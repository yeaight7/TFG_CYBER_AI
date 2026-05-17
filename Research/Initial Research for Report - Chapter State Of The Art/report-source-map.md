# Master Research Map

Below is a research map tuned to your specific setup (flow-based, QRDQN, Gymnasium dataset‑as‑env, CICIDS2017, offline phases), and written so Codex can later turn it into a structured Spanish chapter.

***

## 1. Taxonomy of relevant literature

### 1.1 Benchmark NIDS datasets and flow feature tools

- **CICIDS2017 and related CIC datasets.**
Sharafaldin et al. introduce the CICIDS2017 dataset and compare it against legacy IDS datasets, arguing that older datasets (DARPA, KDD, etc.) are outdated and lack traffic diversity, realistic attack profiles, and modern protocols.[^1][^2]
A follow‑up paper performs a detailed analysis of CICIDS2017, including feature importance and Random Forest baselines using the generated labelled flows.[^3][^4]
An evaluation framework and “reliable dataset” criteria by the same group define design principles (realism, labeled traffic, diversity), useful for justifying dataset selection and discussing limitations.[^5][^6]
The CICFlowMeter tool documentation and Tor/VPN characterization papers define and justify the flow feature set (80+ bidirectional flow features) used for CIC datasets.[^7][^8][^9]
- **Other NIDS datasets and dataset surveys.**
UNSW‑NB15’s official description details its hybrid generation in a cyber range lab, attack taxonomy, and flow-level feature design, giving you a contrasting modern dataset to reference.[^10][^11]
A recent systematic survey of 89 public NIDS datasets catalogs limitations (class imbalance, outdated traffic, missing metadata), and provides a taxonomy and quality criteria directly relevant to arguing about dataset bias and external validity.[^12]
Zenodo resources linking labeled CICIDS2017/UNSW‑NB15 PCAPs to payload‑byte representations show how others reconstruct labeled traffic from PCAP + CSV, which is relevant for your Phase 2 lab traffic pipeline.[^13]

**Why this group matters:** It supports your **dataset selection**, your **flow-based representation (CICFlowMeter features)**, the **limitations section** (e.g., class imbalance, non‑stationarity, realism), and an eventual **external validation discussion** by contrasting CICIDS2017 with UNSW‑NB15 and other datasets.

***

### 1.2 Flow-based NIDS and ML/DL methods

- **Flow‑based IDS techniques and challenges.**
A classic survey on flow‑based intrusion detection reviews architectures, NetFlow‑style features, and statistical vs. ML techniques, and highlights that ML has been under‑exploited compared to rule‑based systems in flow‑based NIDS.[^14]
Flow‑based anomaly detection works in SDN contexts show concrete ML pipelines on flow aggregates, including traffic flow rate prediction and anomaly detection tasks.[^15]
- **ML and DL for NIDS (general).**
Surveys on ML‑based IDS for IoT and general network environments provide taxonomies of algorithms, features, and deployment models, and stress usual metrics (accuracy, precision, recall, F1, FPR, FNR).[^16][^17][^18]
Multiple systematic reviews of deep learning‑based NIDS (general and IoT‑specific) document architectures (AE, CNN, RNN/LSTM, hybrid models) and note that CICIDS2017, CSE‑CIC‑IDS2018, CICDDoS2019, NSL‑KDD, and UNSW‑NB15 dominate benchmarking; they also criticize frequent over‑optimistic results and lack of generalization testing.[^19][^20][^21][^22]
- **Benchmarking on CICIDS2017.**
Sharafaldin et al.’s detailed analysis paper includes Random Forest baselines and “superfeatures” derived via dimensionality reduction; RF performs very well, making it a natural supervised baseline for your thesis.[^4][^3]
Additional benchmarking works on CICIDS2017 (e.g., recent arXiv evaluations and comparative studies) re‑confirm that classical ensembles like RF remain strong baselines versus deep models on tabular flow features.[^23][^24][^25]

**Why this group matters:** This literature underpins **supervised baselines (Random Forest, etc.)**, **metric choices**, and your **positioning of RL vs. strong ML/DL baselines**, avoiding strawman comparisons.

***

### 1.3 Reinforcement learning fundamentals and extensions (DQN, Double/Dueling, distributional, QRDQN)

- **Value‑based deep RL foundations.**
DQN achieves human‑level control on Atari using a CNN + Q‑learning, experience replay and target networks, and is the canonical reference for value‑based deep RL.[^26][^27][^28]
Double DQN shows that standard DQN significantly overestimates Q‑values and that decoupling action selection and evaluation improves stability and Atari performance.[^29][^30][^31]
Dueling networks introduce a value/advantage decomposition that improves performance, especially when many actions have similar value, by sharing state‑value estimation across actions.[^32][^33]
- **Distributional RL and QRDQN.**
Bellemare et al. introduce the distributional RL perspective and C51, modelling the full return distribution rather than only its expectation, and provide convergence analysis in the categorical projection setting.[^34][^35][^36]
Dabney et al.’s “Distributional RL with Quantile Regression” (QR‑DQN) models the return distribution via quantile regression, yielding state‑of‑the‑art Atari performance and enabling risk‑sensitive policies through distributional statistics.[^37][^38][^39][^40]
Later work analyzes quantile‑based methods through the Cramér distance and non‑crossing constraints, providing a theoretical justification for QR‑DQN‑style algorithms and their gradient properties.[^41][^42]
Multi‑step off‑policy distributional RL analyses derive theoretical guarantees and algorithms like QR‑DQN‑Retrace, documenting how distributional TD errors differ from value‑based TD errors.[^43][^44]
- **Applications of distributional RL.**
Recent applications of QR‑DQN and related distributional RL algorithms to risk‑averse trading, condition‑based maintenance, and robotics illustrate practical benefits of distributional estimates for risk‑sensitive control and safety‑critical decision‑making.[^45][^46][^47][^34]

**Why this group matters:** These sources justify your **choice of QRDQN**, support a **risk‑aware interpretation of the return distribution**, and give you **language for reward design and cost‑sensitive decision‑making** (e.g., focusing on CVaR of false‑negative‑weighted returns).

***

### 1.4 RL for NIDS and autonomous cyber defence

- **DRL‑based NIDS surveys.**
Recent surveys devoted to deep RL for NIDS and IoT NIDS provide taxonomies of DRL algorithms (DQN variants, actor–critic, hybrid approaches) applied to intrusion detection, document datasets (CICIDS2017, NSL‑KDD, UNSW‑NB15, etc.), and highlight open issues such as class imbalance, minority/unknown attack detection, and training efficiency.[^48][^49][^50]
- **Autonomous cyber defence (ACD).**
Work on autonomous cyber defence using deep RL (including quantum‑inspired extensions) discusses defender agents that react to attacker agents in simulated networks, focusing on sample efficiency, safety, and training stability.[^51][^52]
These environments are often more abstract than flow‑level NIDS (e.g., Markovian models of attack graphs), but they provide conceptual support for framing your defender as a **sequential decision‑maker with asymmetric costs**.

**Why this group matters:** This literature is your **State of the Art for RL‑based cyber defence**, clarifying what is novel about a **flow‑level, dataset‑as‑environment QRDQN defender** and what remains open (e.g., active online blocking).

***

### 1.5 Evaluation methodology, cross-dataset generalization, and external validation

- **Dataset evaluation frameworks.**
The CIC group propose an evaluation framework for IDS datasets, emphasizing criteria such as attack diversity, traffic realism, labeling quality, and feature coverage; they also argue that many older datasets are unreliable.[^6][^2][^1][^5]
A systematic survey of NIDS datasets formalizes limitations and proposes selection criteria, including specific notes on CICIDS2017 and UNSW‑NB15.[^12]
- **Cross‑domain and long‑term evaluation.**
Cross‑domain studies evaluate ML‑based NIDS across four modern datasets and show that models with near‑perfect in‑dataset accuracy often generalize poorly when trained on one dataset and tested on another, with significant asymmetry depending on source/target choice.[^53]
Another study assesses long‑term performance by training on older datasets and testing on newer ones, showing that decision trees and random forests can overfit heavily to a specific dataset, whereas SVM and shallow ANNs are somewhat more robust.[^54]
Evaluation methodologies tailored to anomaly‑based NIDS further demonstrate substantial performance drops when realistic deployment scenarios and tailored test distributions are considered.[^55]
- **External validation datasets and cyber ranges.**
UNSW‑NB15 and related cyber‑range datasets are generated in controlled lab environments and explicitly intended for evaluating IDS in realistic network settings.[^11][^10]
The VizSec dataset collection and ARCS/OpTC corpora list enterprise and cyber‑range datasets useful as candidates for future external validation beyond CICIDS2017.[^11]

**Why this group matters:** This literature supports your **experimental protocol** choices (strict splits, cross‑dataset checks), your **external validation design** (lab traffic, cyber range analogy), and your **critique of over‑optimistic in‑dataset results**.

***

### 1.6 Cost-sensitive IDS, FP/FN trade-offs, and reward design

- **Cost models for IDS reactions.**
Classical work on cost‑sensitive IDS introduces optimization models that trade off false positive cost vs. intrusion damage cost, identifying optimal operating ranges rather than single point metrics.[^56][^57][^58]
More recent work explores cost matrices and spread‑subsampling for NSL‑KDD, quantifying how changing the relative cost of FNs vs. FPs affects NIDS performance.[^59]
- **Cost‑sensitive ML for NIDS.**
CSE‑IDS proposes cost‑sensitive deep learning and ensemble methods for handling class imbalance in NIDS, leveraging explicit cost terms and reweighting techniques.[^60]
General cost‑sensitive learning and class‑imbalance surveys provide algorithmic options (reweighting, resampling, cost‑sensitive SVM/NN) relevant if you want to align RL rewards with a cost matrix.[^60]

**Why this group matters:** These works are the backbone for **reward function design** and **evaluation of FP/FN trade‑offs**; you can explicitly tie your reward shaping (e.g., negative reward proportional to FN cost) to this literature.

***

### 1.7 Reproducibility, tools, and dataset-as-environment formulations

- **Reproducible ML‑based IDS.**
Recent work proposes reproducible baselines for ML‑based IDS, emphasizing clear experimental protocols, implementation details, and sharing code/configurations.[^61]
Mininet‑IDS is a command‑line framework that connects dataset preprocessing, model training, and evaluation in Mininet, explicitly branding itself as a step toward reproducible ML‑IDS research; it uses NSL‑KDD but the methodology is transferable.[^62]
- **Dataset‑as‑environment and offline RL.**
The Gymnasium project defines a standard RL API and is widely used in both online and offline RL benchmarks, including dataset‑backed environments where `env.get_dataset()` returns transitions for offline training.[^63][^64]
DSRL and related offline RL benchmark suites provide environments where fixed datasets are exposed through a Gym‑like interface, illustrating a pattern similar to your “dataset‑as‑environment” design.[^65][^64]
- **SB3‑contrib and QRDQN implementation.**
The `sb3-contrib` package documents QR‑DQN as an experimental but widely used implementation built on Stable‑Baselines3, intended for research and offering a standard interface for hyperparameters, logging, and seeding.[^66][^67]

**Why this group matters:** These sources justify your **use of Gymnasium and SB3‑contrib**, your **dataset‑as‑environment formulation**, and your **reproducibility and experiment‑management practice** (multiple seeds, code release, clear protocols).

***

### 1.8 Methodological risks: adversarial robustness and over-optimistic metrics

- **Adversarial attacks against DL‑based NIDS.**
Recent work proposes poisoning and evasion frameworks targeting deep NIDS models and demonstrates significant vulnerability on several CIC‑based datasets.[^68]
Surveys of adversarial NIDS catalog attack and defense mechanisms, highlighting that DL models with high nominal accuracy can still be fragile to small perturbations.[^69]
- **Over‑optimistic metrics and lack of realism.**
Surveys and dataset studies note that many NIDS papers report near‑perfect accuracy (up to 99.99%) on public datasets, often with unclear splits or pre‑filtered data; tables summarizing IIoT NIDS show multiple such high scores explicitly accompanied by caveats about limited dataset detail and scalability.[^21][^70][^19]

**Why this group matters:** This literature is crucial for your **limitations and future work** section (adversarial robustness, evaluation biases) and for your **“garbage filter”** criteria regarding suspicious results.

***

## 2. Source matrix

Below is a compact but dense matrix of key sources. For Codex, treat this as the canonical index and extend it as needed. (Many columns contain short phrases by design.)

> Abbreviations:
> Type = survey / dataset / method / benchmark / framework / methodology / standards.
> Task = binary NIDS / multi‑class NIDS / RL control / dataset design / evaluation, etc.
> Metrics/Protocol are summarized; details are in the cited papers.

### 2.1 Core sources table

| ID | Citation key | Full reference | Year | Type | Main topic | Secondary topic | Algorithm / system / dataset | Dataset(s) used | Task | Metrics | Experimental protocol | Code? | Main contribution | Weaknesses | Evidence quality | Relevance | Thesis chapter | Link / DOI |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| S1 | Sharafaldin2018-ICISSP | Sharafaldin, Lashkari, Ghorbani, “Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization,” ICISSP.[^1][^2] | 2018 | dataset / methodology | CICIDS2017 design | Dataset limitations | CICIDS2017 | CICIDS2017 | Dataset creation, baseline ML | Acc, DR, FPR, etc. | Traffic capture over 5 days; labeled flows via CICFlowMeter; train/test splits for baselines | No public code beyond dataset tools | Defines CICIDS2017 and argues for its realism vs. older datasets | Limited external validation; lab environment only | High | Essential | Background, datasets, limitations | [Springer/SCITEPRESS link in ] |
| S2 | Sharafaldin2018-Analysis | Sharafaldin et al., “A Detailed Analysis of the CICIDS2017 Data Set,” IFIP/ICISSP chapter.[^3][^4] | 2018 | benchmark / methodology | CICIDS2017 analysis | RF baseline | CICIDS2017, RF, superfeatures | CICIDS2017 | Binary \& multi‑class NIDS | Acc, Precision, Recall, F1 | Feature selection, superfeatures; RF vs. other ML; cross‑validation | Likely scripts, not fully open | Detailed EDA \& ML baselines on CICIDS2017 including RF | Still single‑dataset; risk of optimistic metrics | High | Essential | Datasets, baselines, evaluation | [^3] |
| S3 | Lashkari2017-CICFlowMeter | Lashkari et al., “Characterization of Tor Traffic Using Time Based Features,” ICISSP, plus CICFlowMeter docs.[^7][^9][^8] | 2017–2025 | framework | Flow feature extraction | Tor/VPN characterization | CICFlowMeter | Tor dataset, CIC traffic | Flow classification | Acc, DR, etc. | PCAP→flows via CICFlowMeter; time‑based features | CICFlowMeter source on GitHub | Defines 80+ bidirectional flow features \& categories | Features tuned to CIC pipelines; not the only possible representation | High | Essential | Feature engineering, methodology | [^9] |
| S4 | Moustafa2015-UNSWNB15 | Moustafa \& Slay, “UNSW‑NB15: a comprehensive data set for network intrusion detection systems,” MilCIS.[^10] | 2015 | dataset | UNSW‑NB15 | Cyber range | UNSW‑NB15 | UNSW‑NB15 | Multi‑class NIDS | Acc, DR, etc. | IXIA‑generated traffic; predefined train/test splits | Dataset tools partly available | Modern dataset with hybrid benign/attack traffic and 49 flow features | Synthetic attacks; protocol mix differs from CICIDS2017 | High | Useful | Datasets, external validation | [^10] |
| S5 | Ring2017-FlowSurvey | Sperotto et al., “Flow-based intrusion detection: Techniques and challenges,” Computers \& Security.[^14] | 2017 | survey | Flow-based NIDS | ML techniques | N/A | NetFlow, IPFIX, etc. | NIDS (various) | Acc, DR, FPR | Survey of prior systems, no unified protocol | N/A | Comprehensive review of flow-based IDS approaches | Pre‑DL era; limited on deep models | High | Essential | Background, flow NIDS | [^14] |
| S6 | Sharafaldin2016-EvalDataset | Sharafaldin et al., “An Evaluation Framework for Intrusion Detection Dataset,” IEEE.[^6] | 2016 | methodology / framework | Dataset evaluation | Criteria design | N/A | Multiple legacy datasets | Dataset assessment | N/A | Proposed criteria; qualitative and quantitative scoring | N/A | Evaluation framework for IDS datasets | Focused on classical datasets; CICIDS2017 comes later | High | Useful | Methodology, limitations | [^6] |
| S7 | Sharafaldin2017-ReliableDataset | Sharafaldin et al., “Towards a Reliable Intrusion Detection Benchmark Dataset,” Journal of Sensor and Networks.[^5] | 2017 | methodology | Dataset design | Limitations of existing datasets | Dataset generation model | Multiple | Dataset assessment | N/A | Survey + design model | N/A | Argues lack of reliable datasets and proposes design/evaluation framework | Does not include all later datasets | High | Useful | Background, limitations | [^5] |
| S8 | DatasetSurvey2025 | “Network intrusion datasets: A survey, limitations, and …,” Computers \& Security.[^12] | 2025 | survey | NIDS datasets | Limitations | N/A | 89 datasets | NIDS | N/A | Systematic literature review | N/A | Large SLR of public NIDS datasets, taxonomy \& limitations | May not cover very latest datasets | High | Essential | Background, dataset selection | [^12] |
| S9 | CrossDomain2023 | “Explainable Cross-domain Evaluation of ML-based Network …,” Computers \& Security.[^53] | 2023 | methodology / benchmark | Cross-domain NIDS | Explainability | 8 ML models | 4 NIDS datasets (incl. CICIDS2017, UNSW-NB15) | Cross-domain NIDS | Acc, F1, FPR, FNR | Train/test on different datasets; SHAP analysis | Not always public code | Shows poor cross-dataset generalization; asymmetry across domains | Limited number of models \& datasets | High | Essential | Evaluation, limitations, external validation | [^53] |
| S10 | EvalLongTerm2022 | “Evaluation of Machine Learning Algorithms in Network …,” arXiv.[^54] | 2022 | methodology | Long-term performance | Overfitting | DT, RF, SVM, ANN, DNN | CICIDS2017, CSE‑CIC‑IDS2018, LUFlow | NIDS | Acc, F1, etc. | Train on earlier dataset; test on later dataset | Possibly code; not central | Demonstrates strong overfitting of DT/RF across datasets/time | Limited to supervised models | Medium | Useful | Limitations, discussion | [^54] |
| S11 | FlowSDN2020 | Satheesh et al., “Flow-based anomaly intrusion detection using machine learning model with software defined networking …,” Microprocessors \& Microsystems.[^15] | 2020 | method | Flow-based IDS | SDN context | ML models | SDN flow datasets | Anomaly detection | Acc, FPR, etc. | Train/test splits on SDN flows | Likely not fully reproducible | Concrete flow-based ML IDS design in SDN | SDN-specific; dataset may not be public | Medium | Optional | Related work | [^15] |
| S12 | DLNIDS-Survey1 | “Deep Learning-Based Network Intrusion Detection Systems: A Comprehensive Survey of Methods, Evolution, and Research Perspectives,” IEEE.[^19] | 2026 | survey | DL-based NIDS | Adversarial challenges | N/A | Many datasets (CICIDS2017, etc.) | NIDS | Acc, DR, FPR, etc. | Survey of 10 years of DL‑NIDS | N/A | Up‑to‑date survey of DL NIDS; discusses dataset issues and limitations | Long; focuses on DL only | High | Essential | State of the Art | [^19] |
| S13 | DLNIDS-SLR2024 | “Deep Learning-Based Network Intrusion Detection Systems: A Systematic Literature Review,” SLR.[^21] | 2024 | survey | DL-based NIDS | Dataset usage | N/A | CICIDS2017, CSE‑CIC‑IDS2018, NSL‑KDD, UNSW‑NB15 | NIDS | Precision, Recall, F1 | PRISMA‑style SLR | N/A | Systematic mapping of DL NIDS, datasets, metrics | Does not cover RL | High | Essential | State of the Art | [^21] |
| S14 | DRL-NIDS-Survey1 | Yang et al., “A Survey for Deep Reinforcement Learning Based Network Intrusion Detection,” journal.[^48][^50] | 2024 | survey | DRL for NIDS | Challenges | DQN, actor–critic variants | CICIDS2017, NSL‑KDD, others | DRL‑based NIDS | Acc, recall, FPR, etc. | Taxonomy of DRL‑NIDS works | N/A | First focused survey on DRL‑based NIDS | Rapidly evolving field; some works may be missing | High | Essential | RL for NIDS, discussion | [^50] |
| S15 | DRL-IoT-NIDS | Gueriani et al., “Deep Reinforcement Learning for Intrusion Detection in IoT: A Survey,” arXiv.[^49] | 2024 | survey | DRL for IoT NIDS | Metrics | DQN, hybrid DRL | IoT datasets | DRL NIDS | Acc, Recall, Precision, FNR, FPR, F-measure | Taxonomy + dataset summary | N/A | Addresses DRL in IoT NIDS with emphasis on metrics and datasets | IoT‑centric; not general networks | High | Useful | RL section, metrics | [^49] |
| S16 | Feng2025-ACD | Feng et al., “Autonomous Cyber Defence by Quantum-Inspired Deep Reinforcement Learning,” ICISSP.[^51][^52] | 2025 | method | Autonomous cyber defence | Training efficiency | QER + QAOA + DQN‑like agents | Simulated network envs | DRL‑based defence | Episodic reward | Multi‑agent training on lab networks | Not flow‑level; heavy quantum components | Shows DRL defenders in ACD scenarios; discusses efficiency vs. complexity | Highly specialized; not directly deployable | Medium | Useful | Background, future work | [^52] |
| S17 | Mnih2015-DQN | Mnih et al., “Human-level control through deep reinforcement learning,” Nature.[^26][^27][^28] | 2015 | method / benchmark | DQN | Atari benchmarks | DQN | Atari 2600 | RL control | Score vs. human | 200M frames, experience replay, target network | Multiple open-source reimpl. | Foundational DQN paper, popularized deep RL | Discrete visual control; no cost-sensitivity | High | Essential | RL fundamentals | [^26] |
| S18 | VanHasselt2016-DoubleDQN | van Hasselt et al., “Deep Reinforcement Learning with Double Q-learning,” AAAI.[^31] | 2016 | method | Double DQN | Overestimation bias | Double DQN | Atari 2600 | RL control | Score | Same protocol as DQN | Many reimpl. | Demonstrates Q overestimation and Double DQN fix | Specific to value-based RL; no distributional view | High | Useful | RL fundamentals | [^31] |
| S19 | Wang2016-Dueling | Wang et al., “Dueling Network Architectures for Deep Reinforcement Learning,” ICML.[^33][^32] | 2016 | method | Dueling networks | Value/advantage decomp | Dueling DQN | Atari 2600 | RL control | Score | Same Atari benchmark | Many reimpl. | Shows value/advantage separation improves performance | Again Atari‑centric | High | Useful | RL fundamentals / future extensions | [^33] |
| S20 | Bellemare2017-Distributional | Bellemare et al., “A Distributional Perspective on RL” (C51).[^35][^36] | 2017 | method | Distributional RL | C51 | C51 | Atari 2600 | RL control | Score, convergence results | Distributional Bellman operator, categorical projection | Reference impl. exists | Introduces distributional RL theory and C51 | Uses categorical support; limited risk‑metrics discussion | High | Essential | Distributional RL | [^35] |
| S21 | Dabney2018-QRDQN | Dabney et al., “Distributional Reinforcement Learning with Quantile Regression,” AAAI.[^39][^38][^37][^40] | 2018 | method | QRDQN | Distributional \& risk-sensitive RL | QRDQN / IQN | Atari 2600 | RL control | Score, risk‑sensitive metrics | Atari benchmark; comparison vs. DQN, C51 | Reference code in DeepMind repos | State‑of‑the‑art distributional DQN; theoretical and empirical results | Atari‑only; hyperparameters tuned for vision tasks | High | Essential | Algorithm choice, RL theory | [^39] |
| S22 | Rowland2021-CramerQR | Rowland et al., “A Cramér Distance perspective on Quantile Regression based Distributional RL.”[^41][^42] | 2021 | method / theory | Distributional RL theory | QRDQN gradients | QRDQN variants | Atari, toy domains | RL control | Score, convergence | Theoretical analysis + experiments | Research code likely | Clarifies metric/gradient aspects of quantile‑based dist. RL | Technical; less directly practical | High | Useful | RL theory section | [^41] |
| S23 | MultiStepDistRL2022 | “The Nature of Temporal Difference Errors in Multi-step Distributional RL,” arXiv.[^44] | 2022 | method / theory | Multi-step distributional RL | Off‑policy learning | QR‑Retrace, etc. | Atari | RL control | Score | Multi‑step off‑policy experiments | Research code | Provides theory for multi‑step Distributional RL | Niche; but relevant for multi‑step updates | Medium | Optional | Advanced RL | [^44] |
| S24 | DistRL-App-CBM2026 | “Distributional RL for Condition-Based Maintenance of Multi-Pump Equipment,” QR‑DQN.[^47] | 2026 | method / application | QRDQN application | Risk-aware maintenance | QRDQN with aging factor | Simulated CBM environment | RL control | ROI, stability, cost | Offline \& online training episodes | Possibly no open code | Shows QRDQN in safety‑critical, cost‑sensitive setting | Domain different from NIDS | Medium | Useful | Reward/risk discussion | [^47] |
| S25 | CSE-IDS-CostSensitive | “CSE-IDS: Using cost-sensitive deep learning and ensemble algorithms to handle class imbalance in network-based intrusion detection systems,” Computers \& Security.[^60] | 2021 | method | Cost-sensitive NIDS | Class imbalance | Cost‑sensitive DL + ensembles | NIDS datasets | NIDS | Acc, F1, FNR | Resampling, cost matrices; standard splits | Unknown | Demonstrates benefit of cost-sensitive methods for imbalanced NIDS | Uses specific architecture; limited generality | Medium | Useful | Reward design, metrics | [^60] |
| S26 | FP-FN-CostModel2014 | “False Positive Responses Optimization for Intrusion Detection System,” journal.[^56] | 2014 | methodology | Cost model for IDS | FP/FN trade-off | Optimization cost model | Synthetic examples | IDS reactions | Cost curves | Simulated scenarios | N/A | Formal cost model relating FP vs. damage cost | Pre‑ML; simplified assumptions | Medium | Useful | Reward design, limitations | [^56] |
| S27 | FP-FN-RiskModel | “Impact of False Positives and False Negatives on Security Risks in …,” risk analysis.[^57] | 2014 | methodology | Security risk modeling | FP/FN | N/A | NSL-KDD or synthetic (depends) | NIDS | Risk metrics | Theoretical model + numerical examples | N/A | Quantifies security risk impact of FP/FN | Dataset may be old (NSL‑KDD) | Medium | Useful | FP/FN discussion | [^57] |
| S28 | CostSensitiveModeling-IDS | Lee et al., “Toward Cost-Sensitive Modeling for Intrusion Detection and Response,” Columbia Tech Report.[^58] | 2000s | methodology | Cost-sensitive IDS | Response selection | Cost-sensitive IDS model | DARPA/KDD | IDS | Cost metrics | Theoretical + early experiments | N/A | Early formalization of cost‑sensitive IDS | Uses outdated datasets | Medium | Useful | Historical background, reward | [^58] |
| S29 | CostSpread-NSLKDD | “Cost and spread value analysis for network intrusion detection …,” thesis.[^59] | 2023 | method | Cost-sensitive learning | Class imbalance | Cost matrix + spread‑subsample | NSL-KDD | NIDS | Acc, F1 | 10‑fold CV + separate test | N/A | Empirical tuning of FN cost and class distribution | On NSL‑KDD only | Medium | Optional | Reward design appendix | [^59] |
| S30 | ReproBaseline-ML-IDS | “Machine Learning for Intrusion Detection: A Reproducible Baseline …,” SmIJ.[^61] | 2024 | methodology | Reproducible ML-IDS | Baseline protocols | ML baselines | NSL-KDD, Edge-IIoTset | NIDS | Standard metrics | Careful experiment design, open configs | Maybe open code | Shows how to structure reproducible ML‑IDS experiments | Older datasets; not CICIDS2017 | Medium | Useful | Methodology, reproducibility | [^61] |
| S31 | Mininet-IDS | “Mininet-IDS: A Step Towards Reproducible Research for Machine Learning Based Intrusion Detection Systems,” IJIST.[^62] | 2024 | framework | Reproducible ML-IDS | Emulation | Mininet-IDS CLI tool | NSL-KDD | NIDS | Acc, F1 | Dataset preprocessing + Mininet deployment | Open-source tool | Tooling for reproducible ML‑IDS experimentation | Uses NSL‑KDD; limited to DDoS | Medium | Optional | Future work, tooling | [^62] |
| S32 | Gymnasium | Gymnasium documentation.[^63] | 2023+ | framework | RL API | Offline RL patterns | Gymnasium | Many envs | RL | Returns, episodic metrics | Standard Gym API, offline datasets via `get_dataset` in some envs | Open-source | Standard RL API; used in offline RL datasets | No specific IDS envs | High | Essential | Methodology (env design) | [^63] |
| S33 | DSRL-OfflineRL | “DSRL: Datasets for Safe Reinforcement Learning,” GitHub.[^65] | 2023 | framework / benchmark | Offline safe RL datasets | Dataset-as-env | DSRL envs | SafetyGym, MetaDrive, etc. | Offline safe RL | Return, cost | Dataset‑as‑environment pattern, `get_dataset()` | Open-source | Shows dataset‑backed Gym env design | Safety‑specific; not NIDS | Medium | Useful | Methodology (dataset-as-env) | [^65] |
| S34 | OfflineRL-Trifinger | “Benchmarking Offline RL on Real-Robot …,” Trifinger datasets.[^64] | 2023 | benchmark | Offline RL robotics | Dataset-as-env | TriFinger RL datasets | Robotics | RL | Returns | Offline RL + option for online policy execution | Open-source | Concrete example of real‑world offline RL with Gym wrappers | Domain unrelated to NIDS | Medium | Optional | Methodology analogue | [^64] |
| S35 | SB3-Contrib-QRDQN | `sb3-contrib` documentation and repo.[^66][^67] | 2020–2025 | framework | QRDQN implementation | Experimental algorithms | QRDQN in SB3 | Atari \& others | RL | Returns | Standard SB3 training loops with QRDQN | Open-source | Provides off‑the‑shelf QRDQN implementation | Labeled “experimental”; defaults tuned for games | High | Essential | Implementation details, methods | [^66] |
| S36 | Adversarial-DL-NIDS | “Poisoning and Evasion: Deep Learning-Based NIDS under Adversarial Attacks,” IEEE.[^68] | 2024 | method / risk | Adversarial attacks on NIDS | CICIDS2017 | Attack framework | CIC-IDS2017, CIC-IDS2018, others | NIDS | Acc, FNR under attack | Poisoning and evasion on DL‑NIDS | Possibly code | Demonstrates DNN‑NIDS vulnerability to adversarial attacks | Focused on DL; no RL | High | Useful | Limitations, security | [^68] |
| S37 | Adversarial-NIDS-Survey | “A Comprehensive Survey of Deep Learning-Based Adversarial Network Intrusion Detection Systems,” IEEE.[^69] | 2025 | survey | Adversarial NIDS | DL models | N/A | Many NIDS datasets | NIDS | Accuracy under attack, etc. | Survey of attacks and defenses | N/A | Summarizes adversarial threats to DL‑NIDS | No RL focus | High | Useful | Limitations, future work | [^69] |
| S38 | Evaluation-SLR-NIDS | “Evaluation of Machine Learning Intrusion Detection Systems …,” methodology (shortcomings).[^55] | 2025 | methodology | NIDS evaluation pitfalls | Anomaly-based NIDS | N/A | Various datasets | NIDS | Detection metrics | Methodology \& performance drop analysis | N/A | Shows how tailored test distributions reveal large performance drops | Focus on anomaly‑based systems | Medium | Useful | Limitations \& methodology | [^55] |

(You can down‑select to ~30 for the “essential bibliography” below; S1–S2, S5, S8–S9, S12–S15, S17–S21, S25–S26, S30–S33, S35–S37 are strong candidates.)

***

## 3. Essential bibliography (30–50 sources)

Below is a curated set grouped by topic; all entries are either directly listed in the matrix or easily added with the same pattern.

### 3.1 NIDS and cybersecurity context

- Sharafaldin2017-ReliableDataset (S7).[^5]
- DatasetSurvey2025 (S8).[^12]
- Ring2017-FlowSurvey (S5).[^14]
- ML‑IDS surveys (IoT and general): at least one comprehensive ML‑based IDS survey.[^17][^18][^16]
- DLNIDS-Survey1 (S12) and DLNIDS-SLR2024 (S13).[^19][^21]


### 3.2 Flow-based detection

- Ring2017-FlowSurvey (S5).[^14]
- FlowSDN2020 (S11).[^15]
- CICFlowMeter / Tor traffic paper (S3).[^9][^7]


### 3.3 Datasets

- Sharafaldin2018-ICISSP (S1).[^1]
- Sharafaldin2018-Analysis (S2).[^3]
- Moustafa2015-UNSWNB15 (S4).[^10]
- VizSec dataset page (UNSW‑NB15, UGR’16, etc.).[^11]
- Zenodo labeled PCAPs for CICIDS2017/UNSW‑NB15.[^13]


### 3.4 Supervised ML/DL baselines

- Sharafaldin2018-Analysis (RF on CICIDS2017) (S2).[^3]
- Additional CICIDS2017 benchmarking work with traditional ML vs DL (e.g., arXiv evaluation or solid comparative study).[^24][^25][^23]
- ReproBaseline-ML-IDS (S30).[^61]


### 3.5 RL fundamentals

- Mnih2015-DQN (S17).[^26]
- VanHasselt2016-DoubleDQN (S18).[^31]
- Wang2016-Dueling (S19).[^33]


### 3.6 Distributional RL / QRDQN

- Bellemare2017-Distributional (C51) (S20).[^35]
- Dabney2018-QRDQN (S21).[^39]
- Rowland2021-CramerQR (S22).[^41]
- MultiStepDistRL2022 (S23) if you need multi‑step details.[^44]
- Application papers using QRDQN in risk‑sensitive domains (S24, plus  as needed).[^46][^47][^45]


### 3.7 RL for NIDS

- DRL-NIDS-Survey1 (S14).[^50]
- DRL-IoT-NIDS (S15).[^49]
- At least one concrete DRL‑NIDS method from these surveys (pick a paper using DQN on CICIDS2017).


### 3.8 Autonomous cyber defence

- Feng2025-ACD (S16).[^52]
- Any additional ACD DRL works cited in DRL-NIDS-Survey1.[^50]


### 3.9 Methodological risks

- CrossDomain2023 (S9).[^53]
- EvalLongTerm2022 (S10).[^54]
- Evaluation-SLR-NIDS (S38).[^55]
- Adversarial-DL-NIDS (S36).[^68]
- Adversarial-NIDS-Survey (S37).[^69]


### 3.10 External validation / cyber ranges

- Moustafa2015-UNSWNB15 (S4).[^10]
- VizSec dataset list and ARCS/OpTC notes.[^11]
- UNSW cyber range description (in S4 + additional UNSW papers).[^10]


### 3.11 Cost-sensitivity and FP/FN trade-offs

- FP-FN-CostModel2014 (S26).[^56]
- FP-FN-RiskModel (S27).[^57]
- CostSensitiveModeling-IDS (S28).[^58]
- CSE-IDS-CostSensitive (S25).[^60]
- CostSpread-NSLKDD (S29).[^59]


### 3.12 Reproducibility and tools

- ReproBaseline-ML-IDS (S30).[^61]
- Mininet-IDS (S31).[^62]
- Gymnasium docs (S32).[^63]
- DSRL \& OfflineRL-Trifinger (S33–S34).[^64][^65]
- SB3-Contrib-QRDQN (S35).[^67][^66]

This set is within the requested 30–50 and covers all thematic axes.

***

## 4. Garbage filter

This section lists **patterns and source types** you should treat skeptically in the thesis, with supporting literature.

### 4.1 Suspiciously high accuracies on public datasets

- **Pattern:** Papers reporting near‑perfect accuracy (e.g., 99.9–99.999%) on CICIDS2017, CSE‑CIC‑IDS2018, UNSW‑NB15, etc., often without mentioning class imbalance or exact splitting strategy. Tables in IIoT/IDS surveys show multiple works with such values and explicitly note limited dataset detail and unclear scalability.[^70][^21][^19]
**Evidence strength:** Moderate (based on meta‑analysis rather than re‑implementation).
**Caveat:** Some methods may genuinely perform very well on specific sub‑tasks (e.g., binary detection of a single dominant attack), but this does not generalize to full multi‑class scenarios.
**Use in thesis:** In **limitations** and **related work**, state that over‑optimistic metrics are common and cite these surveys as evidence.


### 4.2 Unclear train/test splits and cross-validation leakage

- **Pattern:** Many NIDS papers do not clearly separate flows by time, host, or scenario; some perform random row‑wise splits that leak near‑duplicate flows across train/test, inflating performance.[^53][^55][^54]
Cross‑domain and long‑term evaluation work explicitly shows large performance drops when strict cross‑dataset or cross‑time evaluation is used instead of random splits.[^54][^53]
**Evidence strength:** Strong (multiple empirical studies).
**Caveat:** Not every paper with random splits is invalid, but such splits should not be interpreted as deployment‑level performance.
**Use in thesis:** Justify your **strict splitting strategy**, and in **garbage filter subsection** mark results that rely on random row splits as over‑optimistic.


### 4.3 Papers without baselines or dataset description

- **Pattern:** Some IDS works evaluate only a novel method, without comparing to standard baselines (RF, SVM, simple MLP), or provide vague dataset descriptions (“CICIDS2017 subset”) without specifying which days, how labels were preprocessed, or how many flows.[^21][^19][^12]
**Evidence strength:** Moderate (survey‑based).
**Caveat:** You may still cite such works for qualitative ideas but not for quantitative performance claims.
**Use in thesis:** Flag as **weak evidence** in “Related work”; avoid using their performance numbers.


### 4.4 NSL-KDD-only studies overclaiming modern relevance

- **Pattern:** Many cost‑sensitive and ML‑IDS methods are demonstrated exclusively on NSL‑KDD, yet claim “state‑of‑the‑art IDS performance” in general.[^59][^12]
Dataset surveys stress that NSL‑KDD is outdated and cannot represent modern traffic; it’s useful for method prototyping but not as a primary benchmark.[^12]
**Evidence strength:** Strong (widely accepted).
**Caveat:** NSL‑KDD is still useful for methodological demonstrations (e.g., cost matrices); you can reuse ideas but must not assume quantitative transfer.
**Use in thesis:** In **reward design** and **cost‑sensitive learning** sections, treat NSL‑KDD‑only results as conceptual, not performance evidence.


### 4.5 Non-peer-reviewed blogs and opaque repositories

- **Pattern:** Blog posts and GitHub repos sometimes present strong claims (e.g., “Random Forest beats deep models on CICIDS2017 with 99.9% accuracy”) without clear protocols.[^25][^23]
**Evidence strength:** Weak, unless corroborated by peer‑reviewed work (e.g., Sharafaldin2018-Analysis).[^3]
**Caveat:** Use such sources for **implementation hints or preprocessing recipes**, not for formal claims.
**Use in thesis:** Either omit, or clearly mark as **secondary/implementation resource**.

***

## 5. Claims bank

Below is a set of thesis‑ready claims. For each, you have: **Claim**, **Citation keys**, **Evidence strength**, **Caveat**, and **Where to use**. The citation keys refer to the matrix; the inline [web:·] citations refer to tool outputs.

### 5.1 Datasets and flow representation

1. **Claim:** Modern benchmark datasets such as CICIDS2017 were explicitly created to address deficiencies of legacy IDS datasets (DARPA, KDD, etc.), including outdated traffic patterns, limited attack diversity, and anonymized payloads.
    - Citation keys: Sharafaldin2018-ICISSP (S1); Sharafaldin2017-ReliableDataset (S7); DatasetSurvey2025 (S8).
    - Evidence strength: **Strong**.[^1][^5][^12]
    - Caveat: CICIDS2017 itself has limitations (class imbalance, limited days, lab environment).
    - Where to use: **Introduction / dataset selection** to justify using CICIDS2017.
2. **Claim:** CICIDS2017 provides labeled benign traffic and seven attack categories captured over five working days, with flow features generated via CICFlowMeter.
    - Citation keys: Sharafaldin2018-ICISSP (S1); Sharafaldin2018-Analysis (S2); Lashkari2017-CICFlowMeter (S3).
    - Evidence strength: **Strong**.[^9][^1][^3]
    - Caveat: Different releases and CSVs may have slightly different feature counts; you must specify which version you use.
    - Where to use: **Datasets chapter**, ensuring consistency with your preprocessing pipeline.
3. **Claim:** CICFlowMeter computes more than 80 bidirectional flow features spanning identification, packet counts, packet length statistics, timing, TCP flags, window sizes, and activity patterns.
    - Citation keys: Lashkari2017-CICFlowMeter (S3).
    - Evidence strength: **Strong**.[^8][^9]
    - Caveat: Feature definitions may change slightly between tool versions; document exact version.
    - Where to use: **Feature engineering** section when describing the canonical flow vector.
4. **Claim:** UNSW‑NB15 is a modern NIDS dataset created in a cyber range lab, with 49 flow features and nine attack types, and is frequently used in cross‑dataset evaluations alongside CICIDS2017.
    - Citation keys: Moustafa2015-UNSWNB15 (S4); DatasetSurvey2025 (S8); CrossDomain2023 (S9).
    - Evidence strength: **Strong**.[^53][^10][^12]
    - Caveat: Attack mix and traffic conditions differ from CICIDS2017; direct performance numbers are not comparable.
    - Where to use: **External validation** and **related datasets** subsections.

### 5.2 ML/DL baselines and evaluation

5. **Claim:** On CICIDS2017, well‑tuned Random Forest models can achieve very high detection performance and often outperform more complex models on tabular flow features.
    - Citation keys: Sharafaldin2018-Analysis (S2); CICIDS2017 benchmarking papers; DLNIDS-SLR2024 (S13).
    - Evidence strength: **Strong** (multiple studies).[^4][^24][^21][^3]
    - Caveat: Many studies use random row‑wise splits; performance may drop under stricter temporal or scenario splits.
    - Where to use: **Baseline design** section to justify RF as your primary supervised baseline.
6. **Claim:** Cross‑dataset evaluations show that ML‑based NIDS with near‑perfect in‑dataset accuracy may generalize poorly to other datasets, with significant asymmetry depending on which dataset is used for training vs. testing.
    - Citation keys: CrossDomain2023 (S9); EvalLongTerm2022 (S10); DatasetSurvey2025 (S8).
    - Evidence strength: **Strong**.[^54][^53][^12]
    - Caveat: Results depend on specific mapping of features and label harmonization; you must carefully align feature spaces when replicating.
    - Where to use: **Experimental design and limitations**, especially cross‑dataset generalization and Phase 2 lab validation.
7. **Claim:** Many DL‑based NIDS papers report extremely high accuracies on public datasets but are criticized for insufficient details on splits, preprocessing, and deployment realism.
    - Citation keys: DLNIDS-Survey1 (S12); DLNIDS-SLR2024 (S13); DatasetSurvey2025 (S8).
    - Evidence strength: **Moderate**.[^70][^19][^21][^12]
    - Caveat: Some criticisms are qualitative; you should avoid naming individual works unless you have verified their methodology.
    - Where to use: **State of the Art critique** and **garbage filter** section.

### 5.3 RL fundamentals and QRDQN rationale

8. **Claim:** DQN introduced deep value‑based RL with experience replay and target networks, achieving human‑level control on Atari games and becoming the de facto baseline for value‑based deep RL.
    - Citation keys: Mnih2015-DQN (S17).
    - Evidence strength: **Strong**.[^27][^26]
    - Caveat: Atari benchmarks are discrete, fully observable tasks; transfer to NIDS requires careful adaptation of state and reward.
    - Where to use: **RL background** section.
9. **Claim:** Standard DQN suffers from overestimation bias in Q‑values, which Double DQN mitigates by decoupling action selection and evaluation in the target computation.
    - Citation keys: VanHasselt2016-DoubleDQN (S18).
    - Evidence strength: **Strong**.[^31]
    - Caveat: Overestimation severity depends on environment noise and reward structure; in binary intrusion decisions the impact should be empirically assessed.
    - Where to use: **Algorithm choice**, possibly motivating comparison between classic DQN and QRDQN.
10. **Claim:** Distributional RL, as introduced by C51, models the full return distribution via a distributional Bellman operator, and has both theoretical convergence guarantees (in specific metrics) and empirical performance benefits.
    - Citation keys: Bellemare2017-Distributional (S20).
    - Evidence strength: **Strong**.[^35]
    - Caveat: Guarantees rely on specific projections and metric choices (e.g., Cramér distance); using quantiles changes some theoretical details.
    - Where to use: **Distributional RL background** and motivation.
11. **Claim:** QRDQN approximates the return distribution via quantile regression, yields state‑of‑the‑art performance on Atari, and naturally supports risk‑sensitive policies by manipulating quantile‑based objectives.
    - Citation keys: Dabney2018-QRDQN (S21); Rowland2021-CramerQR (S22).
    - Evidence strength: **Strong**.[^38][^37][^39][^41]
    - Caveat: Most experiments focus on maximizing expected return; risk‑sensitive objectives must be explicitly defined.
    - Where to use: **Algorithm choice** and **reward/risk design** sections.
12. **Claim:** QRDQN and related distributional RL algorithms have been successfully applied in safety‑critical and risk‑sensitive domains such as condition‑based maintenance and trading, showing improved performance and controllable risk aversion compared to non‑distributional baselines.
    - Citation keys: DistRL-App-CBM2026 (S24); distributional trading paper.[^47][^45]
    - Evidence strength: **Moderate** (per‑domain).
    - Caveat: Domains differ from cybersecurity; positive results suggest, but do not guarantee, similar benefits.
    - Where to use: **Design rationale** for using QRDQN over standard DQN.
13. **Claim:** The SB3‑contrib library provides a reference implementation of QRDQN with standard Gym‑compatible interfaces, intended for experimental research rather than production deployment.
    - Citation keys: SB3-Contrib-QRDQN (S35).
    - Evidence strength: **Strong**.[^66][^67]
    - Caveat: Marked “experimental”; you should report library version, environment seed handling, and hyperparameters.
    - Where to use: **Implementation details** section.

### 5.4 RL for NIDS and cyber defence

14. **Claim:** DRL‑based NIDS research is still emerging; surveys report that DRL models sometimes outperform traditional DL on specific public datasets, but face unresolved challenges including training efficiency, minority/unknown attack detection, and evaluation under realistic conditions.
    - Citation keys: DRL-NIDS-Survey1 (S14); DRL-IoT-NIDS (S15).
    - Evidence strength: **Moderate**.[^49][^50]
    - Caveat: Surveyed works are heterogeneous and often evaluated on NSL‑KDD or CICIDS2017 with limited reproducibility.
    - Where to use: **State of the Art (RL for NIDS)** and **future work**.
15. **Claim:** Autonomous cyber defence frameworks using DRL demonstrate that defender agents can learn complex response strategies against simulated attackers, but they are computationally demanding and often rely on abstract network models rather than real flow features.
    - Citation keys: Feng2025-ACD (S16).
    - Evidence strength: **Moderate**.[^51][^52]
    - Caveat: These works target training efficiency and exploration (e.g., quantum‑inspired replay) more than classification accuracy; not directly comparable to NIDS tasks.
    - Where to use: **Broader context** section, distinguishing your flow‑based decision setting.

### 5.5 Evaluation, cross-dataset generalization, external validation

16. **Claim:** No ML‑based NIDS model evaluated in recent cross‑domain studies was able to generalize well across all tested datasets; performance varies significantly with the training/target domain pairing.
    - Citation keys: CrossDomain2023 (S9); DatasetSurvey2025 (S8).
    - Evidence strength: **Strong**.[^53][^12]
    - Caveat: Generalization may improve with careful domain adaptation or representation learning; these were not always explored.
    - Where to use: **Limitations** and **justification for Phase 2 lab validation**.
17. **Claim:** Anomaly‑based NIDS suffer large performance drops when evaluated on more realistic test distributions, e.g., with non‑stationary traffic or different attack mixes than the training dataset.
    - Citation keys: Evaluation-SLR-NIDS (S38).
    - Evidence strength: **Moderate**.[^55]
    - Caveat: Specific performance drop magnitudes depend on dataset and method; avoid over‑generalizing.
    - Where to use: **Methodology critique**, to support robust evaluation design.
18. **Claim:** Cyber range datasets like UNSW‑NB15 aim to approximate realistic network conditions in a lab, combining benign traffic and synthetic attacks, and are recommended for testing IDS in near‑realistic scenarios.
    - Citation keys: Moustafa2015-UNSWNB15 (S4); VizSec datasets (S8/S4).
    - Evidence strength: **Strong**.[^10][^11]
    - Caveat: Still controlled; not equivalent to internet‑scale production networks.
    - Where to use: **External validation** and as analogies for your private lab traffic.

### 5.6 Cost-sensitivity, FP/FN trade-offs, reward design

19. **Claim:** Cost‑sensitive IDS models formalize false positives and false negatives as components of an overall operational cost, leading to optimal operating regions rather than a single “best” threshold.
    - Citation keys: FP-FN-CostModel2014 (S26); CostSensitiveModeling-IDS (S28); FP-FN-RiskModel (S27).
    - Evidence strength: **Strong**.[^57][^58][^56]
    - Caveat: Many analyses use simplified assumptions (e.g., fixed damage cost per intrusion); you must discuss how you approximate these in your reward.
    - Where to use: **Reward design** section and **false positive/false negative analysis**.
20. **Claim:** In class‑imbalanced NIDS scenarios, cost‑sensitive training and rebalancing techniques can significantly improve detection of minority attack classes at the expense of some false positives.
    - Citation keys: CSE-IDS-CostSensitive (S25); CostSpread-NSLKDD (S29).
    - Evidence strength: **Moderate**.[^60][^59]
    - Caveat: Many results are on NSL‑KDD; transfer to CICIDS2017 needs empirical confirmation.
    - Where to use: **Reward shaping** and potential **class‑specific penalties** in your RL environment.
21. **Claim:** Designing an RL reward that approximates a cost‑sensitive objective (e.g., high negative reward for false negatives, moderate penalty for false positives) is consistent with cost‑sensitive IDS literature and can be justified as an approximation of operational risk.
    - Citation keys: FP-FN-CostModel2014 (S26); CostSensitiveModeling-IDS (S28); DistRL-App-CBM2026 (S24).
    - Evidence strength: **Moderate**.[^47][^58][^56]
    - Caveat: Actual cost ratios are context‑dependent; you should treat your reward weights as a scenario‑specific assumption and perform sensitivity analysis.
    - Where to use: **Reward design** chapter and **experiments on cost weighting**.

### 5.7 Reproducibility and tooling

22. **Claim:** Reproducible ML‑IDS research requires clear dataset preprocessing pipelines, fixed splits, multiple random seeds, and ideally open code/configuration artifacts; recent work in ML‑IDS demonstrates this with NSL‑KDD and IIoT datasets.
    - Citation keys: ReproBaseline-ML-IDS (S30); Mininet-IDS (S31).
    - Evidence strength: **Moderate**.[^62][^61]
    - Caveat: Existing frameworks may not yet support CICIDS2017 out‑of‑the‑box.
    - Where to use: **Methodology** and **reproducibility** subsections.
23. **Claim:** Gymnasium defines a standard RL API used in both online and offline RL benchmarks; offline RL benchmarks commonly expose fixed datasets via Gym environments using methods like `get_dataset()`.
    - Citation keys: Gymnasium (S32); DSRL-OfflineRL (S33); OfflineRL-Trifinger (S34).
    - Evidence strength: **Strong**.[^65][^64][^63]
    - Caveat: There is no canonical dataset‑as‑environment pattern for classification; your environment design is an adaptation from control literature.
    - Where to use: **Methodology** when defining the Gymnasium dataset‑as‑environment.
24. **Claim:** Implementing QRDQN using SB3‑contrib on top of Gymnasium is aligned with current RL practice; SB3‑contrib is explicitly meant for experimental algorithms, with documented but evolving APIs.
    - Citation keys: SB3-Contrib-QRDQN (S35).
    - Evidence strength: **Strong**.[^67][^66]
    - Caveat: Library updates may change default hyperparameters or behaviors; you should fix a specific version in the thesis.
    - Where to use: **Implementation details**, including version pinning and seed management.

***

## 6. Codex handoff

This section is a guide for the future coding/writing agent (Codex‑style) that will generate the Spanish thesis chapter.

### 6.1 Which sources to cite where

- **Introduction and background (NIDS, datasets, flow features).**
    - Cite S5, S7–S8 (flow‑based detection and dataset surveys) for general IDS and dataset limitations.[^5][^14][^12]
    - Cite S1–S4, S3 for CICIDS2017 and flow features.[^9][^1][^3][^10]
- **State of the Art (ML/DL NIDS).**
    - Cite S12–S13 for DL‑based NIDS surveys.[^19][^21]
    - Cite S2, S9–S10, S36–S37 for CICIDS2017 benchmarks, cross‑dataset performance, and adversarial issues.[^68][^69][^3][^54][^53]
- **RL background and algorithm choice.**
    - Cite S17–S21 for DQN, Double DQN, Dueling, C51, and QRDQN as the theoretical and empirical foundations for your QRDQN choice.[^39][^33][^26][^35][^31]
    - Cite S22–S23 for more advanced distributional RL concepts if needed.[^44][^41]
- **Methodology (dataset-as-environment, implementation).**
    - Cite Gymnasium (S32) and offline RL patterns (S33–S34) when describing the environment design.[^64][^63][^65]
    - Cite SB3-Contrib-QRDQN (S35) for implementation details.[^66][^67]
    - Cite ReproBaseline-ML-IDS (S30) and Mininet-IDS (S31) for reproducibility practices.[^61][^62]
- **Reward design and cost sensitivity.**
    - Cite S25–S29 for cost‑sensitive NIDS and FP/FN cost modeling.[^58][^56][^57][^59][^60]
    - Cite S24 (and ) to connect distributional RL and risk‑sensitive control.[^45][^47]
- **Experimental design and evaluation.**
    - Cite S6–S8 for dataset evaluation frameworks.[^6][^5][^12]
    - Cite S9–S10, S38 for cross‑dataset and long‑term evaluation pitfalls.[^55][^54][^53]
- **RL for NIDS and cyber defence.**
    - Cite S14–S15 for DRL‑based NIDS surveys.[^49][^50]
    - Cite S16 for autonomous cyber defence context.[^52]
- **Limitations and future work.**
    - Cite S9–S10, S36–S37, S38, S8 for generalization, adversarial vulnerability, and dataset limitations.[^69][^68][^55][^12][^54][^53]


### 6.2 Safe vs. risky claims

- **Safe claims (high evidence):**
    - Descriptive properties of CICIDS2017 and UNSW‑NB15 (design, features, attack types).[^1][^10]
    - The fact that RF is a strong baseline on CICIDS2017 with existing evidence.[^3]
    - Theoretical properties and high‑level benefits of distributional RL and QRDQN.[^39][^35]
    - Wide recognition of dataset limitations, over‑optimistic metrics, and poor cross‑dataset generalization.[^12][^54][^53]
- **Moderately safe claims:**
    - That QRDQN can support risk‑sensitive policies via distributional statistics (requires careful definition).[^47][^39]
    - That DRL‑based NIDS can match or surpass DL in some scenarios (reported but heterogeneous).[^50][^49]
    - That cost‑sensitive design improves minority attack detection (e.g., CSE‑IDS).[^60]
- **Risky claims (avoid or qualify heavily):**
    - Any numerical claim of **state‑of‑the‑art accuracy** on CICIDS2017 unless you reproduce it with strict splits.
    - Claims that RL or QRDQN “solve” dataset shift or adversarial robustness; literature does not support this.
    - Claims that NSL‑KDD or CICIDS2017 results generalize directly to all real networks.

Codex should always phrase such statements with qualifiers (“in the evaluated setting”, “under the considered split”, “in our experiments”).

### 6.3 Sections that can be drafted directly

Codex can safely draft, in Spanish, the following sections using the map above:

- **Context and motivation:**
    - Describe NIDS, flow‑based detection, and dataset issues, citing S5, S7–S8, S1–S4.[^5][^14][^1][^10][^12]
- **Related work:**
    - **ML/DL NIDS:** Summaries and critiques based on S12–S13, S2, S9–S10.[^21][^19][^3][^53]
    - **RL for NIDS and ACD:** Based on S14–S16.[^52][^49][^50]
- **Technical preliminaries:**
    - RL, DQN, Double/Dueling, distributional RL, QRDQN based on S17–S21, S22.[^33][^41][^26][^35][^31][^39]
- **Methodology:**
    - Dataset description (CICIDS2017, UNSW‑NB15), feature vector construction (CICFlowMeter), Gymnasium dataset‑as‑env design, QRDQN configuration, supervised baseline design (RF), evaluation metrics and splits (including multiple seeds) using the listed sources.
- **Reward design justification:**
    - Using S25–S29 \& S26–S28 to motivate cost‑sensitive rewards and FP/FN weighting.[^56][^57][^58][^59][^60]
- **Limitations and future work:**
    - Cross‑dataset generalization, adversarial robustness, reproducibility, and external validation using S8–S10, S36–S38.[^68][^69][^55][^54][^53][^12]


### 6.4 Facts needing verification before final LaTeX writing

Before finalizing the thesis, Codex (or you) should explicitly verify:

- **Exact dataset statistics:**
    - Number of flows, class counts, and feature list for your specific CICIDS2017 preprocessed version (e.g., from the original UNB site vs. HuggingFace mirror).[^2][^1]
- **Implementation details:**
    - Exact SB3‑contrib and Gymnasium versions, QRDQN hyperparameters, and seeding strategy (documented in your code, not just in SB3 docs).[^63][^66]
- **Train/validation/test splits:**
    - Ensure the thesis text matches the actual splitting logic (by day, attack type, host, etc.), and that any external lab traffic split is clearly specified.
- **Specific numerical claims:**
    - All numerical results (accuracy, FPR, FNR, AUROC, etc.) must come from your own experiments; literature numbers should only appear for contextual comparison and always with citations and caveats.
- **Any DRL‑NIDS performance claims pulled from surveys:**
    - Check original papers for experimental protocols if you rely on their reported metrics; otherwise keep references qualitative.[^49][^50]

If Codex encounters missing details (e.g., unknown DOIs, ambiguous dataset variants), it should either omit the detail or mark it as “por completar” rather than fabricating.

***

You can now use this map as the backbone for structuring the thesis; each section above corresponds to one or more Spanish subsections, and the claims bank provides pre‑vetted statements with explicit evidence strength and caveats for safe reuse.
<span style="display:none">[^71][^72][^73][^74][^75][^76][^77][^78][^79][^80][^81][^82][^83][^84][^85][^86][^87][^88]</span>

<div align="center">⁂</div>

[^1]: http://www.scitepress.org/DigitalLibrary/Link.aspx?doi=10.5220/0006639801080116

[^2]: https://huggingface.co/datasets/bencorn/CICIDS2017

[^3]: http://link.springer.com/10.1007/978-3-030-25109-3_9

[^4]: https://www.semanticscholar.org/paper/A-Detailed-Analysis-of-the-CICIDS2017-Data-Set-Sharafaldin-Lashkari/cafa09df1905ec46f5a0ab25c2daa77252ed458d

[^5]: http://www.riverpublishers.com/journal_read_html_article.php?j=JSN/2017/1/009

[^6]: http://ieeexplore.ieee.org/document/7885840/

[^7]: http://www.scitepress.org/DigitalLibrary/Link.aspx?doi=10.5220/0006105602530262

[^8]: https://deepwiki.com/ahlashkari/CICFlowMeter/4-feature-extraction

[^9]: https://www.unb.ca/cic/research/applications.html

[^10]: https://research.unsw.edu.au/projects/unsw-nb15-dataset

[^11]: https://vizsec.org/data/

[^12]: https://www.sciencedirect.com/science/article/abs/pii/S0167404825001993

[^13]: https://zenodo.org/records/7258579

[^14]: https://www.sciencedirect.com/science/article/abs/pii/S0167404817301165

[^15]: https://www.semanticscholar.org/paper/Flow-based-anomaly-intrusion-detection-using-model-Satheesh-Rathnamma/3faddd8ef81fcebd65af3b4483bb413c5fd7b00f

[^16]: https://link.springer.com/10.1007/s44163-025-00578-1

[^17]: https://dl.acm.org/doi/10.1016/j.comnet.2019.01.023

[^18]: https://discovery.ucl.ac.uk/10190465/1/Machine-Learning-based-Intrusion-Detection-Systems-compressed.pdf

[^19]: https://ieeexplore.ieee.org/document/11412845/

[^20]: https://ieeexplore.ieee.org/document/10959996/

[^21]: https://pure.uj.ac.za/en/publications/deep-learning-based-network-intrusion-detection-systems-a-systema/

[^22]: https://www.themoonlight.io/en/review/deep-learning-based-intrusion-detection-systems-a-survey

[^23]: https://josep-audenis.github.io/posts/2025/09/dl-vs-ml/

[^24]: https://arxiv.org/pdf/2506.19877.pdf

[^25]: https://github.com/noushinpervez/Intrusion-Detection-CICIDS2017

[^26]: https://www.nature.com/articles/nature14236

[^27]: https://github.com/epignatelli/human-level-control-through-deep-reinforcement-learning

[^28]: https://tomrochette.com/machine-learning/papers/volodymyr-mnih-human-level-control-through-deep-reinforcement-learning/

[^29]: https://hadovanhasselt.com/tag/double-q-learning/

[^30]: https://www.emergentmind.com/topics/double-q-learning-ddqn

[^31]: https://arxiv.org/abs/1509.06461

[^32]: https://pemami4911.github.io/paper-summaries/deep-rl/2016/01/28/dueling-networks.html

[^33]: https://proceedings.mlr.press/v48/wangf16.html

[^34]: https://par.nsf.gov/servlets/purl/10273822

[^35]: https://www.maths.ox.ac.uk/node/34457

[^36]: https://adityauser.github.io/posts/2019/06/C51/

[^37]: https://www.semanticscholar.org/paper/d85623ffae865f9ef386644dd02d0ea2d6a8c8de

[^38]: https://ojs.aaai.org/index.php/AAAI/article/view/11791

[^39]: https://arxiv.org/abs/1710.10044

[^40]: https://aaai.org/papers/11791-distributional-reinforcement-learning-with-quantile-regression/

[^41]: https://www.semanticscholar.org/paper/2cdbddb14304434aef9fdb3d22e04fb89a742330

[^42]: https://www.semanticscholar.org/paper/8fa167d0db69e90b376b608acf534a640ff3d870

[^43]: https://openreview.net/pdf?id=C8Ltz08PtBp

[^44]: https://arxiv.org/abs/2207.07570

[^45]: https://arxiv.org/abs/2501.04421

[^46]: https://linkinghub.elsevier.com/retrieve/pii/S221501612500398X

[^47]: https://arxiv.org/abs/2602.00051

[^48]: https://arxiv.org/abs/2410.07612

[^49]: https://arxiv.org/abs/2405.20038

[^50]: https://onlinelibrary.wiley.com/doi/10.1002/ail2.70026

[^51]: https://www.scitepress.org/publishedPapers/2025/131518/pdf/index.html

[^52]: https://orca.cardiff.ac.uk/id/eprint/176590/

[^53]: https://www.sciencedirect.com/science/article/abs/pii/S0045790623001167

[^54]: https://arxiv.org/abs/2203.05232

[^55]: https://uca.hal.science/hal-05191179/file/nids_tailoring_2.pdf

[^56]: https://www.scirp.org/journal/paperinformation?paperid=43038

[^57]: https://repository.londonmet.ac.uk/6776/1/VV-28 - Impact of False Negatoives and False Positives.pdf

[^58]: https://ids.cs.columbia.edu/sites/default/files/wenke-acmccs2k-cost.pdf

[^59]: https://ujcontent.uj.ac.za/esploro/outputs/graduate/Cost-and-spread-value-analysis-for/9961204707691

[^60]: https://www.sciencedirect.com/science/article/abs/pii/S0167404821003230

[^61]: https://ojs.sciencesforce.com/index.php/smij/article/view/268

[^62]: https://journal.50sea.com/index.php/IJIST/article/view/1082

[^63]: https://gymnasium.farama.org/index.html

[^64]: https://sites.google.com/view/benchmarking-offline-rl-real

[^65]: https://github.com/liuzuxin/DSRL

[^66]: https://pypi.org/project/sb3-contrib/

[^67]: https://github.com/Stable-Baselines-Team/stable-baselines3-contrib

[^68]: https://ieeexplore.ieee.org/document/10788064/

[^69]: https://ieeexplore.ieee.org/document/11089572/

[^70]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12655908/table/sensors-25-06958-t001/

[^71]: https://ieeexplore.ieee.org/document/8888419/

[^72]: https://ieeexplore.ieee.org/document/8585560/

[^73]: https://inria.hal.science/hal-04317161/document

[^74]: https://github.com/nycu-hsl/improving-generalization-of-ML-based-IDS

[^75]: https://link.springer.com/10.1007/s10207-024-00896-y

[^76]: https://peerj.com/articles/cs-1648

[^77]: https://arxiv.org/abs/2505.11551

[^78]: https://ruja.ujaen.es/items/24f750f8-1995-4c09-9f83-8d7a05aabd59

[^79]: https://www.semanticscholar.org/paper/8f2d7fe4f7164db778b0eaec44580ff52e4fe662

[^80]: https://www.semanticscholar.org/paper/98dd4ad4a61313b39192ff845b4945876a79853f

[^81]: https://www.semanticscholar.org/paper/234ba0b5e18d6505136e15bb845ad20e3677904b

[^82]: https://arxiv.org/abs/2507.08196

[^83]: https://ieeexplore.ieee.org/document/9877913/

[^84]: https://www.semanticscholar.org/paper/36f4917876b173071f1b16c610546d0434a6b2fa

[^85]: https://www.semanticscholar.org/paper/cc43349d4fd3676ec299c9a9613da39f4392f995

[^86]: https://www.slideshare.net/slideshow/dueling-network-architectures-for-deep-reinforcement-learning/63832938

[^87]: https://www.neuralaspect.com/posts/breakout-2015

[^88]: https://research.google/pubs/distributional-reinforcement-learning-with-linear-function-approximation/

