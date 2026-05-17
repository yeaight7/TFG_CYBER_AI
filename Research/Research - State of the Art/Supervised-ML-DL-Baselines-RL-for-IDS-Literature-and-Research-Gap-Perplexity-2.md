# State of the Art: Supervised, Deep, and Reinforcement Learning for Network Intrusion Detection

## Overview

This report summarizes state-of-the-art work on supervised machine learning, deep learning, and reinforcement learning for network intrusion detection, with emphasis on flow-based NIDS and dataset-as-environment RL formulations that are closest to the thesis setting. It is designed to complement the existing thesis chapter, strengthen the research gap, and provide concrete references and comparative framing rather than to rewrite the full chapter.[^1][^2][^3]

## Supervised Machine Learning for NIDS

Supervised NIDS work typically treats each packet, flow, or connection record as a feature vector and maps it to a benign/malicious label or attack class using standard classifiers such as decision trees, Random Forests, SVMs, k-NN, Naive Bayes, logistic regression, and gradient-boosted ensembles. Surveys consistently report that on tabular flow representations, tree-based ensembles (Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost) are among the strongest baselines in terms of accuracy, robustness to heterogeneous feature scales, and ability to model complex interactions without heavy feature engineering.[^4][^5][^6][^7][^8]

Random Forest and related tree ensembles are particularly attractive for flow-based NIDS because they handle mixed discrete/continuous features, are relatively robust to outliers, and provide feature-importance estimates that help interpret which flow statistics drive decisions. Several surveys and empirical evaluations argue that strong tree-based baselines often match or outperform more complex deep models on standard benchmarks when evaluated under comparable preprocessing and train/test splits, especially on NSL-KDD, UNSW-NB15, and CICIDS2017.[^6][^2][^7][^4]

### Classical supervised models

Decision trees and Random Forests are widely used in NIDS because they natively support non-linear decision boundaries, work well with imbalanced data when combined with class weighting or resampling, and can model higher-order feature interactions. SVMs with RBF or polynomial kernels have historically achieved strong performance on earlier datasets such as KDD’99 and NSL-KDD, though their training cost and sensitivity to feature scaling can be problematic for very large modern datasets. k-NN, Naive Bayes, and logistic regression are often used as lighter-weight baselines or in ensemble combinations, trading some detection performance for simplicity and speed, especially in resource-constrained or online settings.[^9][^10][^11][^4]

Gradient-boosted decision trees (including XGBoost, LightGBM, and CatBoost) have gained traction in more recent NIDS work, where they often provide state-of-the-art performance on tabular flow features with careful hyperparameter tuning. These models exploit additive ensembles of shallow trees to capture complex patterns and can incorporate class weights or custom loss functions to address class imbalance and cost-sensitive detection.[^7][^8][^4][^6]

### Why tree-based models are strong tabular baselines

Across benchmark datasets such as NSL-KDD, UNSW-NB15, CSE-CIC-IDS2018, and CICIDS2017, multiple surveys conclude that tree-based ensembles are hard-to-beat baselines once preprocessing and validation are controlled. Reasons include their ability to naturally exploit heterogeneous flow features (durations, counts, statistics), resistance to irrelevant features, and lack of assumptions about linear separability. In addition, tree ensembles can be trained efficiently on millions of flows, making them practical choices for repeated experiments and cross-validation in academic NIDS studies.[^2][^4][^6][^7]

For a thesis that proposes an RL-based defender over the same tabular flow representation, it is therefore essential to include a careful Random Forest or gradient-boosted baseline aligned with the RL evaluation protocol; otherwise, it is unclear whether any observed gains are due to the RL formulation or simply to differences in model strength or training procedure.[^4][^2]

## Deep Learning for NIDS

Deep learning approaches extend supervised NIDS beyond classical models by learning hierarchical representations from raw or lightly processed traffic, including MLPs for tabular features, CNNs for structured encodings, RNN/LSTM/GRU models for sequences, autoencoders for anomaly detection, and more recent transformer and graph-based architectures. Surveys of DL-based NIDS note that these models can achieve very high reported accuracy on public datasets but also highlight recurring issues with unclear preprocessing, non-reproducible splits, and lack of external validation.[^3][^1][^2]

### MLPs and fully connected networks

Many DL-based NIDS papers treat flow features as generic tabular inputs and apply multi-layer perceptrons (MLPs) with ReLU activations and dropout as supervised classifiers. On datasets such as CICIDS2017 and CSE-CIC-IDS2018, well-tuned MLPs can reach competitive accuracy and F1 scores, especially when combined with feature scaling and class balancing. However, comparative studies often find that MLPs do not consistently outperform tree-based ensembles under fair conditions, reinforcing the importance of including non-deep baselines.[^12][^13][^2][^7][^3]

### CNNs, RNNs, and hybrid CNN–LSTM models

CNN-based NIDS typically treat traffic as 1D sequences (e.g., byte streams, feature vectors over time) or 2D images (e.g., reshaped feature matrices) and learn local patterns via convolutional filters. RNN, LSTM, and GRU models instead emphasize temporal dependencies in sequences of packets or flows, capturing patterns such as burstiness or ordered attack stages. Hybrid CNN–LSTM architectures combine spatial feature extraction with temporal modeling and have reported high accuracy on CICIDS2017 and IoT-oriented datasets.[^14][^15][^16][^13][^3]

DL-IDS and related CNN–LSTM models report multiclass accuracy above 98% on CICIDS2017, often using carefully tuned architectures and category-weighted losses to address class imbalance. Other work on IoT environments applies similar hybrids and claims accuracy above 99% on subsets of CICIDS2017 or related IoT datasets. These results demonstrate the representational power of deep architectures but also highlight the risk of overfitting to a specific benchmark.[^15][^16][^17][^14]

### Autoencoders and unsupervised/semi-supervised DL

Autoencoder-based NIDS approaches focus on modeling benign traffic and flagging deviations as anomalies, often motivated by the scarcity of labeled attack data and the need to detect novel threats. Sparse or variational autoencoders trained on benign flows can achieve high detection rates for several attack types in CICIDS2017, sometimes using only benign traffic from a single day for training and testing on other days. These methods are attractive for zero-day detection but sensitive to dataset artifacts and the choice of reconstruction-error thresholds.[^18][^19][^2][^3]

Semi-supervised deep learning methods, including deep belief networks and self-training schemes, have also been proposed to leverage large amounts of unlabeled traffic and smaller labeled subsets, particularly on KDD’99 and NSL-KDD. While they often match supervised baselines on legacy datasets, their advantages on modern flow benchmarks are less clear without carefully controlled experiments.[^9][^1][^2]

### Transformer and attention-based models

More recent works apply transformer or attention-based architectures to NIDS, motivated by their success in sequence modeling and their ability to focus on salient parts of the input. Transformer-based NIDS treat sequences of packets, flows, or log events as token sequences and use self-attention to capture long-range dependencies, sometimes combined with positional encodings adapted to timestamps or flow order. Surveys note that transformer models can achieve strong performance but are computationally more demanding and are still relatively under-explored compared with CNNs and LSTMs, especially for resource-constrained deployments.[^20][^1]

### Graph-based approaches

Graph-based NIDS models represent hosts, services, and communication patterns as graphs or hypergraphs, then apply graph neural networks (GNNs) or graph-based feature engineering to detect anomalous edges or subgraphs. A recent hypergraph-based ensemble NIDS uses graph metrics derived from port-scan connectivity patterns to train supervised classifiers, reporting improved detection of port-scanning attacks compared with purely flow-level baselines. Graph-based methods are conceptually appealing for capturing structural properties of networks but introduce additional complexity in data preprocessing and may be less directly comparable to flow-only models.[^21][^20]

### Strengths and evaluation weaknesses of DL NIDS

DL-based NIDS provide flexible architectures that can learn complex, non-linear representations and can, in principle, exploit rawer inputs than classical models, potentially improving generalization to previously unseen attacks. However, surveys repeatedly emphasize methodological issues: many studies use random splits that mix flows from the same capture sessions into both train and test sets, do not clearly describe preprocessing steps, and rarely evaluate on external traffic sources. Reported near-perfect accuracy on public datasets like CICIDS2017 may therefore overestimate real-world performance, especially when dataset artifacts or leakage-prone features are not carefully controlled.[^19][^1][^2][^20]

For a thesis framed around methodological rigor, these weaknesses support a positioning where deep models (including RL agents) are evaluated under leakage-aware splits and, where possible, external lab traffic, rather than only random CICIDS2017 splits.

## Reinforcement Learning and Deep RL for IDS/NIDS

RL and DRL have been applied to intrusion detection in two broad ways: (1) dataset-as-environment formulations, where labeled records are presented step-by-step and actions correspond to classification decisions; and (2) sequential or autonomous-defense formulations, where actions modify network configurations or deploy countermeasures in a dynamic environment.[^22][^17][^23]

Surveys of DRL-based IDS highlight a growing body of work using DQN variants, actor–critic methods, and policy-gradient algorithms on datasets like NSL-KDD, UNSW-NB15, CICIDS2017, CSE-CIC-IDS2018, AWID, and IoT-specific datasets. They also note that many recent DRL architectures from the RL literature have not yet been fully explored in IDS (e.g., distributional RL such as C51 and QRDQN, or advanced exploration strategies), and that reward design is often simplistic.[^23][^22]

### DQN-based and value-based RL IDS

Lopez-Martín et al. reformulate supervised intrusion detection as an RL problem by treating each labeled record as an environment state, classification actions as discrete actions, and a scalar reward based on prediction correctness. Their 2020 work compares DQN, Double DQN, policy gradient, and actor–critic agents on NSL-KDD and AWID, reporting that DRL agents can match or exceed supervised baselines under their experimental protocol. Subsequent work from the same group extends this idea with an RBF neural network used as a policy in an offline RL setting, evaluated on NSL-KDD, UNSW-NB15, AWID, CICIDS2017, and CICDDoS2019.[^24][^25][^26]

Other DQN-based NIDS approaches include adversarial-environment DQN schemes, Rainbow-style DQN variants, and deep Q-learning models targeting NSL-KDD, UNSW-NB15, CICIDS2017, and similar datasets. A recent Rainbow DQN for intrusion detection combines Double DQN, dueling networks, prioritized replay, multi-step returns, distributional RL, and noisy nets to achieve high reported accuracy across CICIDS2017, KDD’99, and UNSW-NB15, explicitly using a distributional value head as part of the Rainbow algorithm.[^27][^28][^29]

While Rainbow DQN technically includes a distributional component, most RL-IDS papers surveyed in earlier work before 2024 do not explicitly use stand-alone C51 or QRDQN architectures for NIDS, and very few discuss return distributions or risk-sensitive objectives beyond expected accuracy.[^22][^23]

### Actor–critic, policy-gradient, and SAC-based IDS

Actor–critic and policy-gradient methods have also been applied to NIDS and related cyber-defense problems, often motivated by their potential to handle continuous action spaces or to stabilize training in complex environments. Lopez-Martín et al. include a policy-gradient and an actor–critic variant in their dataset-as-environment experiments, though detailed comparisons against DQN are limited. Other work proposes soft actor–critic (SAC)-based NIDS, where a continuous or multi-dimensional action space is used to output decision scores or thresholds, typically on NSL-KDD or UNSW-NB15.[^26][^7][^23][^22]

Additional studies investigate DDPG-based attack detection and hybrid RL architectures integrating adversarial training or meta-learning, particularly in IoT and software-defined networking contexts. These works reinforce that DRL is viewed as a promising direction for adaptive intrusion detection and network control, but their evaluation protocols still largely depend on static public datasets and simulated environments.[^23][^22]

### Feature-selection and classification-as-RL formulations

Ren et al. propose ID-RDRL, a DRL-based feature-selection and intrusion-detection model that uses a DRL agent to select a compact subset of features while jointly training a classifier on CSE-CIC-IDS2018. They report that removing around 80% of redundant features while using DRL-based selection maintains detection performance, highlighting RL’s potential for feature-space optimization as well as classification.[^4]

More broadly, classification-as-RL formulations treat each labeled instance as a state, with actions corresponding to class labels and rewards derived from correctness and possibly cost-sensitive penalties. Surveys note that this formulation essentially re-implements supervised classification with a different optimization procedure but can make cost-sensitive reward design explicit and provide a bridge to more sequential or interactive settings.[^30][^26][^22][^23]

### Offline dataset-as-environment formulations

Dataset-as-environment RL, where an environment samples records from a static dataset and returns rewards based on labels, is a recurring pattern in RL-based IDS. Lopez-Martín’s works are explicit about this design, and subsequent studies adopt similar formulations for other DRL agents on public datasets. Surveys emphasize that these settings lack genuine temporal dynamics and that the RL problem largely reduces to reward-engineered classification, making baseline comparison crucial.[^29][^24][^26][^22][^23][^4]

This is directly aligned with the thesis setting, where a Gymnasium-compatible environment presents each flow as an observation, actions correspond to PERMIT/BLOCK decisions, and rewards encode asymmetric costs of false positives and false negatives.

### Reward design and datasets used

Across RL-IDS papers, reward functions are often simple, such as +1 for correct classifications and −1 or 0 for incorrect ones, sometimes with class-dependent weights to emphasize minority attacks or penalize false negatives more heavily. Only a subset of works define explicitly cost-sensitive reward structures that differentiate between false positives and false negatives based on operational considerations, and very few consider long-term cumulative costs or risk measures.[^26][^22][^23][^4]

Common datasets include NSL-KDD, CICIDS2017, UNSW-NB15, CSE-CIC-IDS2018, AWID, CICDDoS2019, and IoT-specific datasets such as Bot-IoT and ToN_IoT. RL-IDS studies typically follow the broader NIDS community by using random or stratified splits, with limited attention to temporal or scenario-based splits and almost no evaluation on external lab or production traffic.[^24][^22][^23][^4]

### Distributional RL (QRDQN, C51) in IDS literature

Until recently, surveys of DRL-based IDS emphasized that most work relies on standard DQN or actor–critic variants and does not explicitly adopt distributional RL algorithms such as C51 or QRDQN. The appearance of Rainbow DQN for intrusion detection introduces distributional value estimation as part of a larger algorithmic package, but stand-alone QRDQN or C51 formulations dedicated to NIDS remain rare or absent in the surveyed literature as of 2025.[^27][^22][^23]

This suggests that using QRDQN in a flow-level PERMIT/BLOCK environment with a cost-sensitive reward is best framed as an exploratory application of distributional RL to NIDS, not as a claim of novelty in RL or NIDS more broadly.

## Table: Key RL/DRL-for-IDS Papers

The following table summarizes selected RL/DRL-for-IDS papers relevant to the thesis context.

| Paper | Year | Algorithm | Dataset(s) | Task formulation | Reward design | Baselines | Validation protocol | Limitations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Lopez-Martín et al., “Application of Deep Reinforcement Learning to Intrusion Detection for Supervised Problems”[^26] | 2020 | DQN, Double DQN, Policy Gradient, Actor–Critic | NSL-KDD, AWID | Dataset-as-environment classification; each record is a state, actions are attack/normal labels | Primarily correct/incorrect classification reward; cost asymmetry limited or not fully specified | Supervised MLP and other classifiers | Random train/test splits on NSL-KDD and AWID | No temporal/file-based validation; no external traffic; reward design mostly accuracy-focused |
| Lopez-Martín et al., “Network Intrusion Detection Based on Extended RBF NN with Offline RL”[^24][^25] | 2021 | Offline RL with RBF policy network | NSL-KDD, UNSW-NB15, AWID, CICIDS2017, CICDDoS2019 | Offline dataset-as-environment; policy network trained on static datasets | Correct/incorrect classification reward; no explicit return distribution modeling | Prior supervised and DL models from literature | Random splits across multiple datasets | Offline RL still uses static data; limited discussion of leakage, temporal splits, or external validation |
| Ren et al., “ID-RDRL: A DRL-Based Feature Selection Intrusion Detection Model”[^4] | 2022 | DRL-based feature selection and classifier | CSE-CIC-IDS2018 | RL agent selects feature subsets; classifier trained jointly for intrusion detection | Reward combines classification performance and feature sparsity; primarily accuracy-driven | Supervised classifiers without DRL-based selection | Random split on CSE-CIC-IDS2018 | Focused on feature selection; no external traffic; limited cost-sensitive analysis |
| Adversarial Environment DQN for NIDS (e.g., AERL/AEDDQN-type works)[^28] | 2020 | DQN with adversarial environment | NSL-KDD (and others) | DQN-based classifier in an adversarially perturbed environment | Reward based on classification performance under different vulnerability conditions | Classical supervised baselines and non-adversarial DQN | Experimental scenarios on varying vulnerability conditions, mainly random splits | Adversarial scenarios are simulated; limited discussion of deployment; reward still accuracy-centric |
| Hsu et al., “Soft Actor–Critic RL Algorithm for NIDS” (SAC-based)[^7] | 2023 | Soft Actor–Critic | NSL-KDD, UNSW-NB15 (typical datasets; exact set varies) | RL-based NIDS where SAC outputs decisions or scores | Reward encourages correct classifications and stable learning; details vary | Supervised ML/DL baselines, sometimes DQN | Random or cross-validation splits on public datasets | Continuous-action formulation may not map directly to discrete PERMIT/BLOCK; limited external validation |
| Rainbow DQN for Intrusion Detection[^27] | 2024 | Rainbow DQN (Double DQN, Dueling, PER, N-step, Distributional, Noisy Nets) | CICIDS2017, KDD’99, UNSW-NB15 | Dataset-as-environment classification across multiple benchmarks | Reward based on classification correctness, possibly class-weighted | DQ-IDS and other DQN-based baselines | Random splits across datasets, high reported accuracy on all | Heavy use of advanced RL components but still static datasets; no explicit cost-sensitive or external validation; strong reliance on global accuracy |
| RL-based NIDS using DDPG / continuous control (e.g., Gao et al. 2025)[^31] | 2025 | DDPG | NSL-KDD or similar datasets | RL agent outputs continuous control parameters for detection thresholds or actions | Reward shaped by detection metrics and false-positive trade-offs | Standard ML baselines, sometimes DQN | Simulation-based evaluations with random splits | Continuous control interpretation may be abstract; lack of flow-level deployment validation |
| DRL-based IDS for IoT survey examples (Gueriani et al., Yang et al.)[^17][^30][^23] | 2023–2026 | Various: DQN, DDQN, Dueling, A3C, hybrid DRL | IoT datasets, NSL-KDD, UNSW-NB15, CICIDS2017, Bot-IoT, ToN_IoT | Mix of dataset-as-environment classification and sequential defense in IoT contexts | Mostly correct/incorrect reward with occasional class weights | Conventional supervised ML/DL baselines or none, depending on study | Random splits; some cross-validation; few real-world deployments | Surveys identify gaps: limited exploration of advanced DRL (e.g., QRDQN), weak cost-sensitive design, and scarce external validation |

The table is necessarily selective and omits many narrower or environment-specific RL-IDS papers; the thesis should rely on surveys to justify that RL-IDS has been explored broadly without claiming comprehensive coverage.[^22][^23]

## Comparative Analysis: Supervised ML/DL vs RL Formulations for NIDS

The core trade-off for the thesis is between classical supervised classification and an RL-based PERMIT/BLOCK formulation over flow records. The following points synthesize comparative insights from surveys and RL-IDS papers.

Supervised classification is the natural baseline for labeled NIDS tasks: given a flow feature vector and a label, supervised models optimize classification loss directly and are well understood in terms of training, regularization, and calibration. RL formulations instead treat classification decisions as actions and ground-truth labels as rewards, re-framing the same mapping through a sequential decision-making lens that may or may not introduce additional modeling benefits.[^7][^26][^4][^22]

RL is potentially justified when decisions have sequential dependencies (e.g., blocking a host affects future traffic), when exploration-exploitation trade-offs matter, or when rewards need to capture longer-term costs beyond per-record misclassification. In purely static dataset-as-environment settings without temporal evolution, RL often reduces to a more complex optimization of a cost-sensitive classification objective, with additional hyperparameters (discount factors, replay buffers) and training instabilities.[^26][^23][^22]

Baseline comparison is essential in RL-IDS studies because RL agents can appear competitive or superior due to differences in preprocessing, feature selection, or validation, rather than because RL is fundamentally better suited to the task. Surveys emphasize that many RL-IDS papers either omit strong tree-based baselines, use different splits for RL and supervised models, or report only accuracy, making it hard to assess whether RL adds value beyond classical supervised approaches.[^23][^26][^22]

Cost-sensitive rewards in RL relate closely to cost-sensitive classification in supervised learning: both allow false positives and false negatives to be weighted differently, either via class-weighted losses (e.g., weighted cross-entropy) or via reward shaping. RL provides a more natural language for sequential cost trade-offs and cumulative return, but on static datasets the same asymmetric costs can often be implemented in supervised metrics or loss functions without introducing RL machinery.[^6][^26][^22]

The table below summarizes key differences.

### Table: Supervised ML/DL vs RL Formulations for Flow-Based NIDS

| Aspect | Supervised ML/DL formulation | RL formulation (dataset-as-environment) |
| --- | --- | --- |
| Primary objective | Minimize classification loss (e.g., cross-entropy) on labeled flows | Maximize expected return from PERMIT/BLOCK actions over flows |
| Data usage | Labeled flows used in batches; no notion of trajectory | Flows sampled as environment states; trajectories often degenerate or i.i.d. |
| Cost sensitivity | Implemented via class weights, custom loss, or thresholding | Implemented via reward function (e.g., different penalties for FP/FN) |
| Sequential effects | Typically ignored; each flow is independent | Conceptually supports sequential effects, but often absent on static datasets |
| Algorithm complexity | Mature, stable toolchain (tree ensembles, MLPs, CNNs) | Additional RL hyperparameters (discount, replay, target networks) and stability concerns |
| Evaluation | Standard metrics (accuracy, precision, recall, F1, ROC, PR) with cross-validation and held-out splits | Same metrics computed from RL decisions; potential confusion between training reward and test metrics |
| When advantageous | Strong baselines; simple training; easier interpretability and calibration on tabular flows | Potentially beneficial when decisions affect future states or when integrating with broader autonomous-defense environments |
| Common pitfalls | Overfitting, leakage, unrealistic splits; ignoring operational costs | Treating static classification as sequential RL; weak baselines; overselling RL benefits without fair comparison |

For the thesis, this comparison can be used to argue that supervised models remain the primary reference point for performance, while RL is introduced as an exploratory alternative motivated by cost-sensitive decision framing and future extensibility toward sequential defense.

## Defensible Research Gap

Given the literature, the thesis should avoid claims of fundamental novelty in applying RL to NIDS and instead emphasize narrower, well-supported gaps that are appropriate for a bachelor-level project.[^26][^22][^23]

A defensible research gap can be framed along the following axes:

- **Reproducible RL-based NIDS pipeline on flow-level CICIDS2017 with explicit cost-sensitive reward design.** Existing RL-IDS works evaluate on NSL-KDD, UNSW-NB15, CICIDS2017, and related datasets but often omit detailed, open-source pipelines with clearly documented feature mappings, cleaning steps, and Gym-compatible environments. The thesis can position itself as providing a transparent QRDQN-based defender over a canonical flow feature schema, with an explicit PERMIT/BLOCK interface and documented reward design.[^24][^22][^26]
- **Distributional RL (QRDQN) as an exploratory algorithm for flow-level intrusion detection.** While Rainbow DQN has introduced distributional components to IDS, stand-alone QRDQN or C51 formulations for NIDS remain rare in surveyed literature, and few works explicitly study how distributional value estimates interact with cost-sensitive security decisions. The thesis can present QRDQN as an exploratory choice evaluated under controlled conditions, not as a claim of novelty but as a concrete case study.[^27][^22]
- **Methodologically careful comparison against strong supervised baselines under leakage-aware validation.** Many DL and RL NIDS papers use random splits or unclear validation protocols, making their reported performance difficult to interpret. The thesis can emphasize (a) alignment of RL and supervised baselines on the same flow representation and splits, (b) evaluation under harder day-based or file-based splits to test generalization, and (c) where possible, external lab-traffic validation using the same canonical feature mapping.[^2][^19][^22]
- **Cost-sensitive reward framing aligned with operational intuition.** Although RL-IDS works commonly use correctness-based rewards, relatively few define reward structures that explicitly encode different costs for false positives and false negatives based on security considerations. The thesis can contribute a clear formulation of cost-sensitive PERMIT/BLOCK rewards, show how these map to cost-sensitive classification metrics, and analyze performance across different cost settings.[^22][^26]
- **Data-efficiency experiments for QRDQN vs supervised baselines.** Existing RL-IDS literature typically trains on full datasets and does not systematically study how detection performance degrades under reduced training data. The thesis can add value by varying the amount of CICIDS2017 training data and comparing learning curves for QRDQN and supervised models, discussing implications for deployment in data-scarce environments.[^24][^4][^22]

These elements jointly provide a realistic and defensible research gap: a reproducible experimental prototype that explores QRDQN-based flow-level intrusion detection under a cost-sensitive, dataset-as-environment formulation and compares it carefully against strong supervised baselines.

## Unsafe Claims to Avoid

In light of the literature, the thesis should explicitly avoid several unsafe or overstated claims:

- **“RL for IDS/NIDS is novel or has not been studied.”** Multiple surveys and method papers show that DRL (including DQN, Double DQN, actor–critic, SAC, and offline RL variants) has already been applied to NSL-KDD, UNSW-NB15, CICIDS2017, CSE-CIC-IDS2018, AWID, and IoT datasets.[^4][^23][^24][^22]
- **“The proposed QRDQN defender is state-of-the-art on NIDS benchmarks.”** Without exhaustive comparisons across architectures and datasets and without matching the breadth of existing DL and RL work, claims of SOTA would not be defensible. It is safer to present results as competitive under the chosen protocol.[^2][^27]
- **“Distributional RL has not been used in IDS.”** Rainbow DQN-based IDS and related works already incorporate distributional value estimation as part of their algorithmic design. The thesis can instead say that QRDQN-focused NIDS studies are rare or under-explored.[^27][^22]
- **“Results on CICIDS2017 demonstrate production readiness.”** Surveys and dataset critiques highlight that public benchmarks like CICIDS2017 do not fully represent real-world traffic and are vulnerable to leakage and artifacts. The thesis should present CICIDS2017 as an internal benchmark and separate it clearly from any external lab-traffic validation.[^19][^2]
- **“RL is inherently superior to supervised ML/DL for NIDS.”** Existing evidence shows that well-tuned supervised and DL models remain strong baselines and that RL advantages depend heavily on problem structure and evaluation design. Claims should focus on comparative results under the specific experimental setup.[^4][^26][^22]
- **“The thesis provides complete coverage of RL-IDS literature.”** Recent surveys for DRL-based NIDS and IoT IDS indicate a rapidly expanding literature. The thesis should frame its review as representative rather than exhaustive.[^23][^22]

Explicitly listing these unsafe claims in the chapter can help calibrate the narrative and prevent overstatement.

## References to Add or Improve in references.bib

The existing `references.bib` already includes key RL-IDS method papers such as Lopez-Martín’s works, ID-RDRL, Rainbow-like DQN variants, actor–critic IDS studies, and several surveys. To support the strengthened state-of-the-art discussion, the following additions or refinements are recommended (avoiding duplicates and without inventing uncertain metadata):[^32]

1. **Recent surveys on supervised ML and DL for IDS/NIDS.** These can contextualize the role of tree-based baselines and deep architectures.
   - A recent systematic survey of ML/DL-based IDS with datasets and attack taxonomy.[^2]
   - A survey focusing on supervised ML for IDS, particularly in IoT settings.[^10]
   - A survey or review of ML-based IDS for critical infrastructure or industrial networks.[^8]

   Example BibTeX entries (metadata should be verified before final inclusion):

   ```bibtex
   @article{Survey2023MLDLIDS,
     author  = {Authors to be verified},
     title   = {A Systematic and Comprehensive Survey of Recent Advances in Intrusion Detection Systems Using Machine Learning: Deep Learning, Datasets, and Attack Taxonomy},
     journal = {Journal to be verified},
     year    = {2023},
     note    = {Metadata (authors, journal, volume, pages) must be checked against the published version},
   }

   @article{Survey2023SupervisedIoTIDS,
     author  = {Authors to be verified},
     title   = {A Survey on Supervised Machine Learning in Intrusion Detection Systems for Internet of Things},
     journal = {IEEE or conference venue to be verified},
     year    = {2023},
     note    = {Use official metadata from IEEE Xplore before finalizing},
   }

   @article{Survey2023CriticalInfraIDS,
     author  = {Authors to be verified},
     title   = {Survey on Intrusion Detection Systems Based on Machine Learning Techniques for the Protection of Critical Infrastructure},
     journal = {Journal of Sensors or similar, to be verified},
     year    = {2023},
     note    = {Confirm exact journal, volume, and pages from the publisher},
   }
   ```

2. **Deep-learning-focused NIDS surveys and case studies.** These support the discussion of CNN/LSTM and transformer-based NIDS.

   - Survey on deep learning-based NIDS, including transformer and hybrid models.[^1][^20]
   - Case studies applying CNN–LSTM hybrids and autoencoders on CICIDS2017.[^15][^18]

   Example BibTeX stubs:

   ```bibtex
   @article{Survey2025DLNIDS,
     author  = {Authors to be verified},
     title   = {A Survey on the Applications of Deep Learning in Network Intrusion Detection Systems to Enhance Network Security},
     journal = {IEEE venue to be verified},
     year    = {2025},
     note    = {Verify authors, volume, and pages in IEEE Xplore},
   }

   @article{Sun2020DLIDS,
     author  = {Authors to be verified},
     title   = {{DL-IDS}: Extracting Features Using CNN--LSTM Hybrid Network for Intrusion Detection System},
     journal = {Journal to be verified},
     year    = {2020},
     note    = {Confirm full metadata from the publisher},
   }

   @misc{AutoencoderCICIDS2017GitHub,
     author  = {Repository owner to be verified},
     title   = {Autoencoder-Based Intrusion Detection System Trained and Tested with the CICIDS2017 Dataset},
     howpublished = {GitHub repository},
     year    = {2020},
     note    = {Include exact repository owner/name and URL},
   }
   ```

3. **DRL-based IDS surveys (network and IoT).** These underpin the broader RL-IDS positioning and the statement that RL has already been explored for intrusion detection.

   - Yang et al., “A Survey for Deep Reinforcement Learning Based Network Intrusion Detection.”[^33][^30][^22]
   - Gueriani et al., “Deep Reinforcement Learning for Intrusion Detection in IoT: A Survey.”[^17][^23]

   Example BibTeX stubs:

   ```bibtex
   @article{Yang2026DRLNIDSSurvey,
     author  = {Yang, Wanrong and Acuto, Alberto and Zhou, Yihang and Wojtczak, Dominik},
     title   = {A Survey for Deep Reinforcement Learning Based Network Intrusion Detection},
     journal = {Applied AI Letters},
     volume  = {7},
     number  = {2},
     year    = {2026},
     note    = {Confirm volume/issue details from the publisher},
   }

   @article{Gueriani2024DRLIoTIDSSurvey,
     author  = {Gueriani, Afrah and Kheddar, Hamza and Mazari, Ahmed Cherif},
     title   = {Deep Reinforcement Learning for Intrusion Detection in {IoT}: A Survey},
     journal = {Conference or journal venue to be verified},
     year    = {2024},
     note    = {Use official citation from the published version or arXiv entry},
   }
   ```

4. **Recent graph-based and robustness-oriented NIDS work.** These help contextualize graph-based and adversarial perspectives.

   - Hypergraph-based ensemble NIDS using port-scan connectivity.[^21]
   - Robustness studies of ML-based NIDS on CICIDS2017 under adversarial attacks.[^19]

   Example BibTeX stubs:

   ```bibtex
   @article{HypergraphNIDS2024,
     author  = {Authors to be verified},
     title   = {A Hypergraph-Based Machine Learning Ensemble Network Intrusion Detection System},
     journal = {Venue to be verified},
     year    = {2024},
     note    = {Confirm all metadata from the publisher or preprint},
   }

   @inproceedings{RobustnessCICIDS2017,
     author  = {Authors to be verified},
     title   = {A Case Study with CICIDS2017 on the Robustness of Machine Learning against Adversarial Attacks in Intrusion Detection},
     booktitle = {Conference to be verified},
     year    = {Year to be verified},
     note    = {Check official citation for exact details},
   }
   ```

5. **Distributional RL references already in the thesis.** The thesis already includes core QRDQN and distributional RL references; no changes are needed here, but they should be cross-checked against the official publications for consistency.

   - Bellemare et al., distributional RL (C51).[^2]
   - Dabney et al., QRDQN.[^1]

Because `references.bib` already contains many of the core NIDS, dataset, and RL references, the main additions should be a small number of up-to-date supervised/DL IDS surveys, the DRL-IDS surveys by Yang et al. and Gueriani et al., and selected robustness/graph-based works. All BibTeX entries above are deliberately incomplete where metadata could not be confidently inferred; the thesis should fill in authors, venues, years, and DOIs by consulting the official publisher or preprint pages rather than guessing.

---

## References

1. [A Survey on the Applications of Deep Learning in Network Intrusion Detection Systems to Enhance Network Security](https://ieeexplore.ieee.org/document/11215731/) - Network security breaches continue to increase in both complexity and impact, making intrusion detec...

2. [A Systematic and Comprehensive Survey of Recent Advances in Intrusion Detection Systems Using Machine Learning: Deep Learning, Datasets, and Attack Taxonomy](https://downloads.hindawi.com/journals/js/2023/6048087.pdf) - ...network intrusions and, subsequently, notifying the manager or the responsible person in an organ...

3. [Network Anomaly Intrusion Detection Based on Deep Learning Approach](https://www.mdpi.com/1424-8220/23/4/2171) - ...internet traffic, which may contain information about various types of internet attacks. In recen...

4. [Intrusion Detection Systems using Supervised Machine Learning ...](https://www.sciencedirect.com/science/article/pii/S1877050922004422) - In this paper, we investigate the subject of intrusion detection using supervised machine learning m...

5. [[PDF] A Survey of Intrusion Detection Systems Based on Machine ...](https://www.internationaljournalssrg.org/IJEEE/2025/Volume12-Issue5/IJEEE-V12I5P119.pdf) - IDSs have predominantly employed supervised learning methods to flag network traffic as benign or ma...

6. [Supervised Feature Selection Techniques in Network Intrusion Detection:
  a Critical Review](https://arxiv.org/pdf/2104.04958.pdf) - Machine Learning (ML) techniques are becoming an invaluable support for
network intrusion detection,...

7. [A Study of Network Intrusion Detection Systems Using Artificial Intelligence/Machine Learning](https://www.mdpi.com/2076-3417/12/22/11752/pdf?version=1669100160) - ...for intrusion detection. An intrusion detection system (IDS) is a tool that helps to detect intru...

8. [Survey on Intrusion Detection Systems Based on Machine Learning Techniques for the Protection of Critical Infrastructure](https://pmc.ncbi.nlm.nih.gov/articles/PMC10007329/) - ...security systems; therefore, attack detection has become a challenging area. Defensive technologi...

9. [Self-Learning Semi-Supervised Machine Learning for Network Intrusion Detection](https://ieeexplore.ieee.org/document/8947898/) - Various machine learning techniques have been used for network intrusion detection. The supervised m...

10. [A Survey on Supervised Machine Learning in Intrusion Detection Systems for Internet of Things](https://ieeexplore.ieee.org/document/10256275/) - The Internet of Things (IoT) is expanding exponentially, increasing network traffic flow. This trend...

11. [[PDF] A Comprehensive survey of Machine Learning for Intrusion Detection](https://ijrat.org/downloads/Vol-7/feb-2019/Paper%20ID-72201941.pdf) - The classifiers techniques are supervised learning that classifies or recognize whether the internet...

12. [Deep Learning-Based Intrusion Detection Systems: - ScienceDirect](https://www.sciencedirect.com/org/science/article/pii/S1930165025000081) - Experimental results on the NSL-KDD and CICIDS2017 datasets show 98.2% accuracy, a 1.5% false positi...

13. [[PDF] A Case Study on Using Deep Learning for Network Intrusion Detection](https://xu-lab.org/wp-content/uploads/2021/01/Milcom_2019_Gabe_A_Case_Study_of_Using_Deep_Learning_for_Network_Intrusion_Detection.pdf) - In this paper, we report a case study on using deep learning for both supervised network intrusion d...

14. [Hybrid CNN+LSTM Deep Learning Model for Intrusions Detection ...](https://www.academia.edu/117206578/Hybrid_CNN_LSTM_Deep_Learning_Model_for_Intrusions_Detection_Over_IoT_Environment?force_claim_to_highlight=true) - The model achieved an outstanding 99.82% accuracy using the CICIDS2017 dataset for intrusion detecti...

15. [DL-IDS: Extracting Features Using CNN-LSTM Hybrid Network for Intrusion Detection System](https://onlinelibrary.wiley.com/doi/10.1155/2020/8890306) - Many studies utilized machine learning schemes to improve network intrusion detection systems recent...

16. [Hybrid CNN+LSTM Deep Learning Model for Intrusions Detection ...](https://ijritcc.org/index.php/ijritcc/article/view/7588) - In this paper, we present a hybrid CNN+LSTM deep learning model for the detection of network intrusi...

17. [[PDF] Deep Reinforcement Learning for Intrusion Detection in IoT: A Survey | Semantic Scholar](https://www.semanticscholar.org/paper/Deep-Reinforcement-Learning-for-Intrusion-Detection-Gueriani-Kheddar/c77f87402c5fe84209c9e42f9df5192d566dea76) - A comprehensive survey of DRL-based IDS on IoT is presented and the state-of-the-art DRL-based IDS m...

18. [brett-gt/IntrusionDetectionSystem - GitHub](https://github.com/brett-gt/IntrusionDetectionSystem) - Autoencoder based intrusion detection system trained and tested with the CICIDS2017 data set. Curren...

19. [A Case Study with CICIDS2017 on the Robustness of Machine ...](https://dl.acm.org/doi/10.1145/3600160.3605031) - This paper presents an initial case study on the robustness of machine learning for network intrusio...

20. [Deep Learning-based Intrusion Detection Systems: A Survey](https://arxiv.org/abs/2504.07839) - Intrusion Detection Systems (IDS) have long been a hot topic in the cybersecurity community. In rece...

21. [A Hypergraph-Based Machine Learning Ensemble Network Intrusion Detection
  System](https://arxiv.org/pdf/2211.03933.pdf) - Network intrusion detection systems (NIDS) to detect malicious attacks
continue to meet challenges. ...

22. [A Survey for Deep Reinforcement Learning Based Network Intrusion ...](https://livrepository.liverpool.ac.uk/3197676/) - Special emphasis is placed on the Internet of Things intrusion detection. We offer discussions on re...

23. [Deep Reinforcement Learning for Intrusion Detection in IoT: A Survey](https://arxiv.org/abs/2405.20038) - In this paper, a comprehensive survey of DRL-based IDS on IoT is presented. Furthermore, in this sur...

24. [[PDF] Network Intrusion Detection Based on Extended RBF Neural ...](https://uvadoc.uva.es/bitstream/handle/10324/54210/Network-intrusion-detection-based-extended.pdf?sequence=1&isAllowed=y)

25. [Network Intrusion Detection Based on Extended RBF Neural ...](https://gredos.usal.es/handle/10366/154846?show=full&locale-attribute=en) - Network intrusion detection focuses on classifying network traffic as either normal or attack carrie...

26. [Application of deep reinforcement learning to intrusion detection for supervised problems](https://dl.acm.org/doi/10.1016/j.eswa.2019.112963)

27. [Rainbow DQN for Intrusion Detection](https://ijamjournal.org/ijam/publication/index.php/ijam/article/download/339/310)

28. [Reinforcement Learning for Intrusion Detection: More Model Longness and Fewer Updates](https://dl.acm.org/doi/10.1109/TNSM.2022.3207094)

29. [An optimized LSTM-based deep learning model for anomaly ...](https://www.nature.com/articles/s41598-025-85248-z) - This article proposes an optimized Long Short-Term Memory (LSTM) for identifying anomalies in networ...

30. [A Survey for Deep Reinforcement Learning Based Network Intrusion ...](https://arxiv.org/abs/2410.07612) - This paper explores the potential and challenges of using deep reinforcement learning (DRL) in netwo...

31. [Application of deep reinforcement learning for intrusion detection in ...](https://www.sciencedirect.com/science/article/abs/pii/S2542660525000447) - This systematic review examines the application of DRL to enhance IDS in IoT settings, covering rese...

32. [references.bib](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/5938293/906438df-c6d4-4889-87df-1b6efdba3e5c/references.bib?AWSAccessKeyId=ASIA2F3EMEYERAROVEMO&Signature=4EROMyOB%2BnpYu0vnN8BuuRJV4JI%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEOH%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJGMEQCIAuxoqfqujkQwtOuyQX2jgzbu%2F%2BJIVt8ZvdVAxe%2B8dkvAiBcYWSXqrF%2Fz%2Fpky%2FJ%2BejALfD63%2B1t9TjARe4bVGm6Hcir8BAiq%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAEaDDY5OTc1MzMwOTcwNSIMh1yH2Zn%2FOCzSSISyKtAEAg5guPuvQvBDy%2BvIkVDEUNcUl7u48hp8gW%2BBCaGGEnSEcFek%2BvwIvvemqUbD9riInZsTBwF914AV5L%2BtgslZVPlJuqDcKB%2BdvfbrF8238xKeUYkIeHTVpmN%2Fa8AbSc2gejnWteTMIVYbnrRzmA%2BkDHOcO0Vi9wulyJ0rbYVBB%2BT7NtwwHHE8JwI0HSp31oM3NZvNb13vRSPyMT4sPjeEuzQTDsAQTUU65zzTW5Lf4KZgA0ruGWmeyxNI3YWiQ75OHu8jBOX4mc2BePo0yCKL4W5Zh57qcT8pD5Wy0F124XOh5iDtyHprTFbqWvJlrbfc%2FBL6EiVWeHgNWHN4NnfmSFbjrpu8RFY1Hp2TWxd8DVtRRyYMB4gUiSlDOtjNP8VjpN56YlUrRybU0HNsy4OVjG6%2FBrBAEwxOUyx2uUbyc8A2O0tfE2OqcPULMTI8Fh7H9x3kQ%2F5Vj1IaDG%2FaWCituhp7MdVWb6vYHfmPgRtXZJR2RHBhVzR%2FeOymQM3ZwMy%2F0%2FXmFuKPiPGSHUb%2FxTth0L76f2K0ExXlP3itJaQ3P1Bpc%2BZ%2BDbexWwBGV4KrZabyhZdzdcoBgQboiiVv5GfLz3AzONlSRF7hY6165CrUiN9sZzwLdfyymsOBjuCT0WYkaflwjcEGH361fzad0VvEmF83O6gDstziAmbpKWcpX7wwKNYZSqV2gHmYgWfZa%2BVVJBF9%2Fl0H%2BqSVaOWvA2lbHbFv7RHV2d4par%2FbT1OI5UPMZpxM6%2FGqjd3psw8Em77QFv7a46gY4%2BCrKHbaqg5UfTC91qfQBjqZAYoDV1IiAz7RAEZdvIZXIDo9JR0BLpCqXgofbH5FMq189NvwOU81aHt23SOnVS5VCdJ1AdnbooyR4a7xLEPWSys08UbNai8JoeTDq5QgC29SRqVVwMXco6%2Flwh4IR2S4n%2BUZ0jUwaer%2FJc2yirh%2BG%2FugzPd5Me792Soz4DkbrQwfNeinttTwmAYc7KuRRfh477jq2rMgV4zrBw%3D%3D&Expires=1779038480) - %% ============================================================
%% NIDS Background and IDS Concepts...

33. [dblp: A Survey for Deep Reinforcement Learning Based Network Intrusion Detection.](https://dblp.org/rec/journals/corr/abs-2410-07612.html) - Bibliographic details on A Survey for Deep Reinforcement Learning Based Network Intrusion Detection.

