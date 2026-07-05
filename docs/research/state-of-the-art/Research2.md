## 1. Short verdict

- Most RL/DRL‑IDS work treats **intrusion detection as supervised classification implemented via RL**, using static datasets (NSL‑KDD, UNSW‑NB15, CICIDS2017, CSE‑CIC‑IDS2018) rather than live traffic.[^1][^2][^3][^4]
- Several influential papers (especially López‑Martín and co‑authors) explicitly adopt a **dataset‑as‑environment formulation**, where each record is a state, the class prediction is the action, and the reward is based on correct/incorrect classification.[^5][^2][^4][^1]
- **DQN and its variants (DDQN, AE‑DQN)** are the most common DRL algorithms; actor‑critic variants (AC, DDPG) and policy‑gradient approaches also appear, but **distributional RL (e.g., QRDQN, C51)** is not used in mainstream IDS literature so far.[^6][^7][^2][^8][^9][^5][^1]
- RL is sometimes used **purely as a classifier or as part of a hybrid pipeline (e.g., feature selection, RBF policy network)** rather than as a long‑horizon decision‑maker; the environment is usually a sampling function over labeled flows.[^2][^4][^1]
- Datasets are mostly **NSL‑KDD, KDDCup’99, CICIDS2017, CSE‑CIC‑IDS2018, UNSW‑NB15, AWID, CICDDoS2019**; newer IoT‑centric works use Bot‑IoT/TON_IoT‑like or IoMT/SDN datasets.[^10][^11][^12][^13][^14][^3][^4][^6][^2]
- Most papers **compare RL/DRL against standard supervised baselines** (SVM, Random Forest, MLP, CNN/LSTM), often reporting modest to strong gains, but evaluation is usually **single‑dataset with random splits and no external validation**.[^15][^16][^4][^10][^6][^1][^2]
- Reward design is often **simple (±1 for correct/incorrect)** or embedded in a loss function; a few recent works introduce **multi‑objective or imbalance‑aware rewards**, but these are exceptions.[^7][^12][^4][^15][^2]
- Methodological issues from supervised ML‑NIDS carry over: **random splits, no temporal validation, no cross‑dataset tests, limited discussion of class imbalance, and lack of released code**, so your thesis can make a strong point by being more rigorous on these aspects.[^4][^8][^9][^17][^10][^6][^1][^2]

***

## 2. Paper matrix

*(Focus on core RL/DRL‑IDS works and those most relevant to a dataset‑as‑environment, binary PERMIT/BLOCK formulation.)*


| ID | Citation key | Paper title | Year | RL algorithm | Dataset | State / observation | Action space | Reward design | Binary or multiclass? | Baselines used | Metrics | Main results | Evaluation split | External validation? | Code? | Methodological concerns | Direct relevance to my project | Link / DOI |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| P1 | LopezMartin2020-DRL-IDS | Application of deep reinforcement learning to intrusion detection for supervised problems (Expert Systems with Applications)[^1][^18][^19] | 2020 | DQN, DDQN, Policy Gradient, Actor–Critic | NSL‑KDD, AWID[^1][^5] | Preprocessed feature vector per flow/connection | Predict one of several attack/normal classes | Reward based on classification error (positive for correct, negative for incorrect); implemented as RL loss over dataset samples[^1][^19][^5] | Multiclass (5 labels on NSL‑KDD) | Several ML/DL baselines (e.g., MLP, SVM, RNN) per text[^1][^5] | Accuracy, F1, recall, precision[^1] | DRL models slightly outperform supervised baselines on NSL‑KDD/AWID | Train/test splits on NSL‑KDD/AWID; appears random (not temporal)[^1] | No | Unknown (repo mentions but not official)[^5] | Uses legacy datasets; random splits; environment is just a sampler; no cross‑dataset generalization | Very high: canonical dataset‑as‑environment DRL classifier, multi‑algorithm, close to your formulation (flow‑vector → action) |  |
| P2 | LopezMartin2021-RBF-OfflineRL | Network Intrusion Detection Based on Extended RBF Neural Network With Offline Reinforcement Learning (IEEE Access)[^13][^4][^20] | 2021 | Offline RL with RBFNN policy (gradient‑based) | NSL‑KDD, UNSW‑NB15, AWID, CICIDS2017, CICDDoS2019[^4] | Flow‑based feature vector (packets aggregated into flows)[^4] | Predicts class (normal vs attacks) | Reward embedded in offline RL loss, optimising classification performance; discusses imbalance impact[^4] | Binary and multiclass (depending on dataset)[^4] | Compares against classical ML and previous DL/RL models[^4] | Accuracy, F1, recall, etc.[^4] | RBF‑RL policy achieves better metrics than baselines across datasets | For each dataset, predefined train/test; looks like random splits, not temporal | Partially: multi‑dataset evaluation but no independent lab traffic | No public code mentioned[^4] | Random/non‑temporal splits; class imbalance handled but not fully cost‑sensitive; limited operational analysis | Very high: multi‑dataset (incl. CICIDS2017, UNSW‑NB15), offline RL classifier on flow features |  |
| P3 | LopezMartin2020-DRL-Supervised | Application of deep reinforcement learning to intrusion detection for supervised problems (conceptual DRL for supervised tasks) – same as P1 but often cited as “conceptual modification of DRL paradigm”[^1][^5] | 2020 | DQN, DDQN, PG, AC | NSL‑KDD, AWID[^1][^5] | Sampled records from dataset | Action = class label | Reward derived from misclassification (e.g., −loss); dataset acts as pseudo‑environment[^1][^19][^5] | Multiclass | Same as P1 | Same | Same | Same | Same | Same | This conceptual point is key for dataset‑as‑environment RL formulation for your thesis |  |  |
| P4 | LopezMartin2021-RBF-Ext | (Same work as P2, but explicitly emphasising imbalance) Network Intrusion Detection Based on Extended RBF Neural Network With Offline RL[^4][^13] | 2021 | Offline RL with RBFNN | NSL‑KDD, UNSW‑NB15, AWID, CICIDS2017, CICDDoS2019[^4] | Flow features | Class | Reward shaped via RL loss with emphasis on minority detection[^4] | Both | Classical ML + DL baselines | Accuracy, F1 | RBF–RL improves metrics, claims robustness on unbalanced data[^4] | Random | No | No | Overlaps with P2; key for class‑imbalance \& RL discussion | High |  |
| P5 | Ren2022-IDRDRL | ID‑RDRL: a deep reinforcement learning‑based feature selection intrusion detection model (Sci Rep)[^2][^3] | 2022 | Deep RL (DRL classifier with RFE feature selection) | CSE‑CIC‑IDS2018[^2][^3] | Reduced feature vector after RFE and NN encoder[^2] | Predicts attack category | Reward based on classification performance; optimises feature subset and classifier jointly[^2][^3] | Multiclass | Traditional ML classifiers and deep NN baselines[^2] | Accuracy, detection rate, FPR, etc.[^2] | Removes ~80% features while maintaining/improving accuracy on CSE‑CIC‑IDS2018[^2] | Train/test split on CSE‑CIC‑IDS2018, appears random | No | Unknown | Single dataset (no cross‑dataset); limited discussion of temporal structure; feature–selection focus | High: DRL for feature selection + classification on CSE‑CIC‑IDS2018; strong CSE‑CIC reference |  |
| P6 | AEDQN2020-NSLKDD | Network Intrusion Detection Systems Using Adversarial Reinforcement Learning with Deep Q‑network (IEEE Access)[^6] | 2020 | AE‑DQN (Adversarial / Multi‑Agent DQN) | NSL‑KDD (KDDTrain+, KDDTest+)[^6] | NSL‑KDD feature vector | Action = 1 of 5 labels (Normal, DoS, Probe, R2L, U2R)[^6] | Likely +1 for correct, −1 for incorrect; adversarial agent perturbs environment; reward not fully detailed in snippet[^6] | Multiclass (5 labels) | Baselines: RNN‑IDS, AESMOTE‑IDS[^6] | Accuracy, macro F1[^6] | ~80% accuracy and 79% macro‑F1 on NSL‑KDD 5‑class classification[^6] | Uses KDDTrain+ for training and KDDTest+ for evaluation (predefined split)[^6] | No | Unknown | NSL‑KDD only; limited analysis of minority classes; reward and environment description are brief | Medium–high: strong DQN‑IDS precedent, but on NSL‑KDD not CICIDS2017 |  |
| P7 | NIDS-RL-2023 | Network Intrusion Detection System using Reinforcement learning (IEEE 2023)[^21] | 2023 | Deep RL (likely AE‑DQN variant) | NSL‑KDD (KDDTrain+/KDDTest+)[^21] | NSL‑KDD feature vector | 5‑label classification | Correct/incorrect classification reward (described qualitatively)[^21] | Multiclass | Compares to RNN‑IDS and AESMOTE variants[^21] | Accuracy, F1, etc.[^21] | ~80% accuracy and 79% macro F1 like AE‑DQN; improved detection for some attacks[^21] | KDDTrain+/KDDTest+ | No | Unknown | Same dataset and general design as AE‑DQN; no external validation; reward design not formalized | Medium: reinforces AE‑DQN line of work; classification‑centric |  |
| P8 | RLTechniques2023 | Network Intrusion Detection System Using Reinforcement Learning Techniques (IEEE 2023)[^10] | 2023 | DQN with distributed agents + attention | NSL‑KDD, CICIDS2017[^10] | Flow / connection features (NSL‑KDD and aggregated flows for CICIDS2017) | Multi‑class intrusion labels | Reward based on correct classification; some discussion of robustness to adversarial perturbations[^10] | Multiclass | Existing IDS (e.g., RNN, autoencoder‑based) as baselines[^10] | Accuracy, precision, recall, F1, FPR[^10] | Claims better accuracy and F1 than baselines and robustness under black‑box attacks[^10] | Dataset‑level train/test splits; likely random for CICIDS2017 (not temporal) | No | Unknown | Uses two datasets but no true external validation; reward not cost‑sensitive; little detail on temporal structure | High: multi‑dataset DRL‑IDS using DQN and CICIDS2017; close to your flow‑based aim |  |
| P9 | Sanusi2023-DRLThesis | Network Intrusion Detection Using Deep Reinforcement Learning (MSc thesis)[^16] | 2023 | DQN | NSL‑KDD (converted to binary)[^16] | NSL‑KDD features per connection | Action = {normal, attack} | Simple +1/−1 reward for correct/incorrect; emphasises adaptivity[^16] | Binary | Traditional ML classifiers as baselines (e.g., MLP, SVM) | Accuracy, F1, loss[^16] | DQN achieves comparable or better accuracy to baselines on binary NSL‑KDD | Random train/test split on NSL‑KDD | No | No | Legacy dataset; random split; no cross‑dataset evaluation | Medium: structurally similar (binary decision), but on NSL‑KDD and without flows or distributional RL |  |
| P10 | DQIDS2025 | Deep Q‑learning intrusion detection system (DQ‑IDS): A novel reinforcement learning approach for adaptive and self‑learning cybersecurity (journal article)[^15][^22][^23][^24] | 2025 | DQN with experience replay, adaptive ε‑greedy | “Real‑world network datasets” (not fully specified; likely NSL‑KDD/CICIDS2017 or similar)[^15] | Flow‑ or connection‑level features per record | Binary (benign vs malicious) | Reward mechanism reinforces correct classifications and penalizes errors to reduce FP/FN[^15] | Binary | Conventional ML and DL IDS models (not fully enumerated) | Accuracy (≈97.18%), F1, etc.[^15] | DQ‑IDS outperforms conventional IDS in detection accuracy and efficiency[^15] | Appears to use standard random train/test splits | No | Unknown | Datasets and splits under‑specified; no cross‑dataset tests; reward described qualitatively, not formally | High: DQN‑based adaptive IDS with simple reward; conceptually close to your PERMIT/BLOCK agent |  |
| P11 | Ren2022-IDRDRL-FeatureSel | ID‑RDRL: DRL-based feature selection intrusion detection model (Sci Rep)[^2][^3] | 2022 | DRL for feature selection + classifier | CSE‑CIC‑IDS2018[^2] | Full → reduced feature vector | Selects feature subset and outputs predicted class | Reward tied to classification performance, penalizing more features and misclassifications[^2] | Multiclass | Classical ML and NN baselines | Accuracy, DR, FPR, feature reduction rate[^2] | Removes 80% features while matching or improving attack detection on CSE‑CIC‑IDS2018[^2] | Random split on CSE‑CIC‑IDS2018 | No | Unknown | Single dataset; no temporal analysis; reward design still primarily classification‑centric | High: DRL on CSE‑CIC‑IDS2018; interesting for feature selection and class imbalance |  |
| P12 | DDPG2025-OptimizedIDS | Optimized Network Security Attack Detection Algorithm Based on Deep Deterministic Policy Gradient (DDPG) (IEEE 2025)[^7] | 2025 | DDPG (continuous‑action actor–critic) | NSL‑KDD, UNSW‑NB15[^7] | Network features per record | Continuous action encoding detection strategy (e.g., thresholding/policies) | Multi‑objective reward balancing detection precision and latency; prioritized replay for class imbalance[^7] | Binary and/or multiclass (depends on experiments) | SVM, autoencoder, DQN, standard DDPG[^7] | Accuracy, F1, AUC[^7] | DDPG variant outperforms baselines on NSL‑KDD and UNSW‑NB15[^7] | Train/test splits on NSL‑KDD and UNSW‑NB15 (likely random) | No | Unknown | Novel reward design but still static datasets; unclear operational interpretation of continuous actions | Medium: shows richer reward and actor–critic, but not flow‑based CICIDS2017 |  |
| P13 | DRL-IDS-SDN2025 | Deep reinforcement learning‑based intrusion detection scheme for SDN (Scientific Reports)[^11] | 2025 | DRL with LFTS‑RNN + optimization (PC‑JTFOA) | NSL‑KDD, WPPD[^11] | SDN traffic features fed to LFTS‑RNN | IDS actions + routing/optimization | DRL reward encourages high sensitivity/specificity and low response time[^11] | Binary / attack vs normal | Compares to several IDS methods | Sensitivity (98.67%), specificity (97.42%), accuracy (99.85%)[^11] | Random split on NSL‑KDD and WPPD | No | Unknown | Very high reported metrics; no external validation; complex hybrid may obscure RL contribution | Medium: interesting for cost‑aware reward, but domain is SDN, not generic flow‑NIDS |  |  |
| P14 | HCLR-IDS2025 | A deep RL‑based robust IDS for securing IoMT healthcare networks (Frontiers in Medicine)[^12] | 2025 | DQN and PPO on top of CNN‑LSTM | CICIoMT2024 (IoMT dataset)[^12] | CNN‑LSTM features from IoMT traffic | Binary/multiclass IDS decision | Reward combines classification performance and exploration; details partially specified[^12] | Both | Several ML/DL baselines | Accuracy, precision, recall, F1[^12] | DRL variants improve detection of evolving threats vs baselines | Random data split on CICIoMT2024 | No | No | New dataset, but same evaluation patterns; RL contribution mostly at decision layer | Low–medium: architecture and reward ideas useful, but dataset/domain differ |  |

*(You can add Caminero’s AE‑RL and AE‑SAC as additional rows; they were covered in the previous answer and are conceptually very close for dataset‑as‑environment DRL, but not all bibliographic details are in this query.)*[^25][^26][^27]

***

## 3. Closest works to my thesis

### 1) López‑Martín 2021 – Extended RBFNN with offline RL (P2/P4)[^13][^4]

- **Why close**: Uses offline RL to train a classifier over flow‑like features, using multiple datasets including CICIDS2017 and UNSW‑NB15; dataset‑as‑environment with emphasis on imbalance.[^4]
- **Similarity**:
    - Treats each flow/connection vector as state and class label as the “action”.
    - Focuses on offline training on public datasets, including CICIDS2017.
- **Differences**:
    - RL is used as an optimization framework for an RBF policy network, not as a Q‑learning agent; no explicit online environment interaction or value function.
    - Task is multi‑class classification, not binary PERMIT/BLOCK.
    - No external validation on lab traffic; evaluation uses random splits.
- **Should you cite as direct precedent?** Yes – this is probably your most important precedent for “offline RL‑style” NIDS, multi‑dataset, flow‑based.


### 2) López‑Martín 2020 – DRL for supervised intrusion detection (P1/P3)[^5][^1]

- **Why close**: Explicitly formulates a **dataset‑as‑environment** DRL classifier, implementing DQN, DDQN, Policy Gradient, and Actor–Critic on NSL‑KDD and AWID.[^1][^5]
- **Similarity**:
    - Each record is a state; action is class prediction; reward is based on correctness.
    - Offline training, RL used to optimise classification behaviour.
- **Differences**:
    - Uses legacy datasets and multi‑class classification; not flow‑based CICIDS2017.
    - No distributional RL; no external validation.
- **Direct precedent?** Yes – it justifies your dataset‑as‑environment Gymnasium design and your use of Q‑learning‑family methods.


### 3) Ren 2022 – ID‑RDRL on CSE‑CIC‑IDS2018 (P5/P11)[^3][^2]

- **Why close**: DRL‑based IDS on **CSE‑CIC‑IDS2018**, combining feature selection and classification with DRL.[^2]
- **Similarity**:
    - Uses a modern CIC dataset with flow‑based features.
    - Embeds reward in classification performance over recorded traffic.
- **Differences**:
    - Focuses on feature selection (RFE + DRL) and classification, not explicit PERMIT/BLOCK decisions.
    - Only one public dataset; no external lab validation; reward not clearly cost‑sensitive.
- **Direct precedent?** Yes – especially for connecting DRL to CSE‑CIC‑IDS2018 and discussing feature selection vs your “fixed canonical feature vector” approach.


### 4) RL Techniques 2023 – NIDS Using RL Techniques (P8)[^10]

- **Why close**: Multi‑agent DQN‑based IDS evaluated on **NSL‑KDD and CICIDS2017**, with emphasis on distributed agents and attention mechanisms.[^10]
- **Similarity**:
    - Uses DQN‑family method; combines flow‑like features with RL.
    - Involves CICIDS2017, which is also your main dataset.
- **Differences**:
    - Multi‑class classification and multi‑agent architecture rather than a single binary agent.
    - Focus on adversarial robustness and attention; no distributional RL; no external validation.
- **Direct precedent?** Yes – strong “DQN + CICIDS2017” baseline to position your QRDQN defender against.


### 5) AE‑DQN 2020 – Adversarial RL with DQN (P6)[^6]

- **Why close**: Uses AE‑DQN to perform 5‑class classification on NSL‑KDD with training on KDDTrain+ and evaluation on KDDTest+.[^6]
- **Similarity**:
    - DQN‑based agent with correct/incorrect reward.
    - Focus on anomaly‑based IDS with RL classification.
- **Differences**:
    - NSL‑KDD only; not flow‑based; multi‑class; adversarial agent modifies environment.
    - No multi‑dataset or external validation.
- **Direct precedent?** Yes – as a characteristic DQN‑IDS, and for discussion of macro‑F1 vs accuracy.


### 6) DQ‑IDS 2025 – Deep Q‑learning IDS (P10)[^23][^15]

- **Why close**: Presents a generic DQN‑based IDS (DQ‑IDS) with experience replay and adaptive exploration; conceptually similar to your QRDQN agent but without distributional value functions.[^15]
- **Similarity**:
    - DQN environment where actions are IDS decisions; reward is correct/incorrect classification (aimed at reducing FP/FN).[^15]
    - Emphasis on adaptivity and self‑learning.
- **Differences**:
    - Dataset choice is underspecified; apparently multiple “real‑world” datasets, but not clearly mapped to CIC/UNSW.
    - No mention of temporal splits or external validation; metrics are high but may suffer from same pitfalls.
- **Direct precedent?** Yes – as a modern DQN‑IDS reference; useful when motivating why you choose a more expressive QRDQN.


### 7) Sanusi 2023 – DQN‑based NIDS (P9)[^16]

- **Why close**: DQN‑based IDS implementing binary classification on NSL‑KDD.[^16]
- **Similarity**:
    - Binary action space (“attack” vs “normal”), similar to PERMIT/BLOCK.
    - Dataset‑as‑environment view, though not formally emphasised.
- **Differences**:
    - NSL‑KDD only; no flow‑based CIC datasets; random split; no external validation.
- **Direct precedent?** Yes – good to cite as an example of binary DQN‑IDS on legacy data.


### 8) ID‑RDRL–Feature Selection (Sci Rep 2022, P11)[^3][^2]

- **Why close**: DRL‑based classifier on CSE‑CIC‑IDS2018 with strong feature‑selection component; uses DRL reward to balance accuracy and feature count.[^2]
- **Similarity**:
    - Uses modern CIC dataset; trains DRL classifier on flow‑like data.
    - Discusses redundant features and dataset imbalance.
- **Differences**:
    - Focuses on feature selection (RFE + DRL), whereas you fix a canonical feature vector.
    - Multi‑class classification; no explicit PERMIT/BLOCK.
- **Direct precedent?** Yes – especially for CSE‑CIC‑IDS2018 and DRL feature selection vs your fixed feature map.


### 9) DDPG 2025 – Optimized attack detection (P12)[^7]

- **Why close**: DDPG‑based IDS with multi‑objective reward balancing detection precision and latency, using NSL‑KDD and UNSW‑NB15.[^7]
- **Similarity**:
    - RL policy that optimizes a cost‑sensitive reward (precision/latency) – relevant for reward design.
    - Includes UNSW‑NB15.
- **Differences**:
    - Continuous action space; abstract “detection strategy” rather than discrete PERMIT/BLOCK.
    - Complex multi‑objective reward; not flow‑specific and no external validation.
- **Direct precedent?** Indirect – good for reward‑design discussion and actor–critic methods, but less directly comparable.


### 10) HCLR‑IDS 2025 \& DRL‑IDS SDN 2025 (P13/P14)[^11][^12]

- **Why close**: Both use DRL (DQN/PPO or DRL + RNN) to build IDS/defense mechanisms for SDN or IoMT.[^12][^11]
- **Similarity**:
    - RL agent making security decisions over network traffic.
    - Some use multi‑objective rewards (detection vs latency vs routing).
- **Differences**:
    - Domain‑specific datasets (IoMT, SDN) rather than general flow datasets; architectures are large hybrids with RL only in some components.
    - No CICIDS2017; no external validation.
- **Direct precedent?** Cite as broader DRL‑IDS examples and for dynamic reward ideas, but they are not your closest technical matches.

***

## 4. Reward design patterns

### Simple correct/incorrect reward

- Many works use a **binary reward**: +1 for correct classification, −1 (or 0) for incorrect classification, with the environment sampling labeled records.[^19][^16][^1][^15]
    - López‑Martín 2020: reward defined by classification error, effectively encouraging correct labels over the dataset.[^19][^5][^1]
    - AE‑DQN, NIDS‑RL 2023: although not fully formalised, the description indicates reward based on overall classification performance on NSL‑KDD (accuracy/F1).[^21][^6]
    - Sanusi 2023: explicitly describes positive reward for correct class and negative for wrong class in DQN on NSL‑KDD.[^16]
    - DQ‑IDS 2025: “reward‑driven training mechanism reinforces correct classifications and penalizes errors,” again matching this simple pattern.[^15]


### Cost‑sensitive reward

- A smaller set incorporates **cost‑sensitivity**, often indirectly:
    - DQ‑IDS 2025 aims to reduce both FP and FN rates through reward shaping (though exact weights are not fully detailed).[^15]
    - DDPG‑based optimized attack detection uses a **multi‑objective reward** balancing detection precision and latency; misclassifications and high latency incur higher penalties.[^7]
    - IoMT and SDN DRL‑IDS (HCLR‑IDS, DRL‑IDS‑SDN) incorporate performance and QoS terms (e.g., response time) into reward alongside detection metrics.[^11][^12]


### Class‑imbalance‑aware reward

- Some RL approaches **acknowledge imbalance** explicitly:
    - Extended RBFNN with offline RL (López‑Martín 2021) analyses class imbalance and suggests that the RL‑optimised RBF policy is particularly suitable for unbalanced datasets; reward/loss implicitly emphasises minority classes but exact weighting is not fully specified.[^4]
    - ID‑RDRL discusses removing 80% of redundant features and improving detection, especially in complex network environments; class imbalance is a motivation, though the reward is still mainly accuracy‑based.[^2]
    - DDPG‑based IDS uses prioritized experience replay “to address class imbalance”, which effectively reweights experiences rather than the scalar reward itself.[^7]


### False‑negative‑heavy penalty

- Some papers informally claim to “reduce false negatives” but only a few clearly state **higher penalties for FN than FP**:
    - DQ‑IDS 2025 emphasises reducing both FP and FN; the text suggests reward discourages especially FN, but numeric weights are not given.[^15]
    - DDPG‑based IDS hints at prioritising detection precision (which is more about FP) and latency; explicit FN‑heavy penalty is not clearly described.[^7]

For your thesis you can introduce an explicitly **FN‑heavy reward** (e.g., higher negative reward for PERMIT+attack vs BLOCK+benign), arguing that most prior work does not formally specify or calibrate this.

### Dynamic/adaptive reward

- Truly **dynamic rewards** (changing over time based on environment feedback) are rare:
    - DDPG‑based IDS uses a multi‑objective reward that might be tuned for latency vs detection during training; however, not clearly adaptive over time.[^7]
    - Hybrid IoMT/SDN works use DRL to adapt policies to evolving traffic, but reward functions themselves are mostly static linear combinations of metrics.[^12][^11]


### Unclear/underspecified reward

- Many RL/DRL‑IDS papers **do not fully specify the reward function**; they only state that RL “maximises detection accuracy” or “reduces misclassification” without giving an explicit formula.[^14][^21][^10][^6]
- This is a methodological weakness you can highlight: RL results cannot be reproduced or interpreted without knowing reward shape and magnitude.

***

## 5. Baselines used in the literature

Across the RL/DRL‑IDS works and surveys:

- **Classical ML baselines**
    - **Random Forest**: commonly used on NSL‑KDD, UNSW‑NB15, CICIDS2017 as a strong tabular baseline.[^8][^1][^4][^2]
    - **SVM**: frequent baseline in NSL‑KDD and UNSW‑NB15 experiments.[^1][^2][^7]
    - **Decision Tree / C4.5**: used as simple, interpretable baseline.[^1][^2]
    - **Logistic Regression, Naïve Bayes, k‑NN**: used in some works and comparative studies.[^28][^1][^2]
- **Neural‑network baselines**
    - **MLP / fully connected DNN**: most common deep baseline.[^28][^4][^1][^2]
    - **CNN, RNN/LSTM, GRU**: widely used for NSL‑KDD, CICIDS2017, UNSW‑NB15; RL works often compare against at least one DL baseline (e.g., RNN‑IDS, CNN‑IDS).[^11][^12][^10][^6]
    - **Autoencoders**: sometimes used as anomaly‑detection baselines or as feature extractors before classical classifiers.[^28][^1][^2]
- **Other baselines**
    - **Ensembles** (XGBoost, gradient boosting, voting classifiers) appear in some DRL‑IDS and feature‑selection works as strong supervised baselines.[^8][^2]
    - **Previous RL methods**: Some works compare a new DRL model against earlier DQN/DRL architectures.[^6][^1][^7]


### Minimum defensible baselines for your thesis

For a bachelor thesis that aims to be methodologically serious:

- At minimum, for **binary PERMIT/BLOCK** on CICIDS2017 (and possibly other datasets), you should have:
    - One **strong classical ML baseline**: e.g., Random Forest or Gradient Boosting on your canonical feature vector.[^28][^1]
    - One **deep supervised baseline**: e.g., a well‑tuned MLP (and optionally a simple CNN if you reshape flows) trained with cross‑entropy.[^29][^28]
- Preferably add:
    - A simpler **logistic regression** baseline as a “sanity check” (especially useful when discussing class imbalance and AUC).[^28]
    - If time allows, one **unsupervised/anomaly baseline** (e.g., autoencoder) to highlight differences in detection philosophy.[^29][^2]

You can then argue that your QRDQN defender is compared against both a classical and a deep supervised baseline, which is common practice in RL‑IDS literature and satisfies basic fairness.[^8][^4][^1][^2]

***

## 6. Weaknesses in the RL/DRL‑IDS literature

These mirror the general ML‑NIDS issues, with RL‑specific twists.

- **Random split problems**
    - Most RL‑IDS works use **random train/test splits** on NSL‑KDD, UNSW‑NB15, CICIDS2017 or CSE‑CIC‑IDS2018, ignoring temporal structure and correlations between flows within the same attack scenario.[^10][^16][^4][^1][^2][^15]
    - This can leak nearly identical flows into both train and test, inflating reported performance; CICIDS2017 and CSE‑CIC‑IDS2018 are especially sensitive to this because of scenario‑based captures.[^30][^31][^32]
- **Lack of external validation**
    - None of the surveyed RL‑IDS papers evaluate on **lab‑captured or real operational traffic** beyond existing public datasets; at best they use multiple public datasets from the same family.[^4][^10][^6][^1][^2]
    - Cross‑dataset generalization (e.g., training on CICIDS2017, testing on UNSW‑NB15 or lab traffic) is absent, despite evidence that ML models generalize poorly across datasets.[^17][^33][^8]
- **Unrealistic or underspecified reward**
    - Many papers define reward only implicitly (“maximise detection accuracy”) without explicit weighting for different errors or operational costs.[^21][^14][^10][^6]
    - Only a few consider latency or resource cost (e.g., DDPG multi‑objective reward in SDN/attack detection).[^11][^7]
- **No temporal validation**
    - RL is almost always applied to static datasets; **no RL‑IDS paper validates in a temporally ordered setting**, e.g., training on earlier days of CICIDS2017 and testing on later days.[^10][^1][^2][^4]
    - This undermines claims about adaptivity to evolving attacks, since the agent never faces “future” distributions during evaluation.
- **No reproducibility / no code**
    - Code is rarely released; a few GitHub projects re‑implement López‑Martín or DRL‑IDS models, but official repositories are uncommon.[^34][^5]
    - Reward definitions, preprocessing pipelines, and train/test splits are often under‑documented, making independent replication difficult.[^1][^2][^4][^15]
- **Inflated metrics and legacy datasets**
    - Some DRL‑IDS works report very high metrics (e.g., >99% accuracy or F1) on NSL‑KDD or KDDCup99, datasets known to be biased and redundant.[^35][^36][^12][^11]
    - Surveys on DRL‑IDS emphasize that many reported gains may be due to dataset peculiarities and random splits.[^9][^17][^8]
- **Class imbalance ignored or superficially treated**
    - Although imbalance is mentioned, many RL‑IDS papers still optimize unweighted accuracy; minority attack detection (e.g., U2R, R2L) remains weak but is not central to evaluation.[^21][^6][^2][^4][^1]
    - DDPG‑based IDS and some hybrid models attempt to address imbalance via prioritized replay or oversampling, but this is not systematic.[^15][^7]
- **No cost‑sensitive evaluation**
    - Very few works report cost‑based metrics (e.g., weighting FN much more heavily than FP) or operational metrics such as number of alerts per minute or analyst workload.[^9][^17][^8][^1]

These weaknesses give you a clear angle: you can design **explicitly cost‑sensitive rewards, temporal or scenario‑based splits, and external validation on lab traffic**, and argue that this is uncommon or missing for RL‑IDS.

***

## 7. Claims I can safely make

Use these as building blocks in your thesis; each should be accompanied by the citation keys and [web:x] citations when you write.

1. **Claim:** Most DRL‑based NIDS treat intrusion detection as a supervised classification problem embedded in an RL framework, often with each dataset record acting as a state and the predicted class as an action.
    - **Supporting sources:** LopezMartin2020‑DRL‑IDS, LopezMartin2021‑RBF‑OfflineRL, Ren2022‑IDRDRL, AE‑DQN2020, NIDS‑RL‑2023.[^21][^6][^2][^4][^1]
    - **Strength of evidence:** Strong; multiple independent papers explicitly describe dataset‑as‑environment formulations.
    - **Caveat:** Some works use RL only for feature selection or as part of larger hybrid models rather than as the main classifier.
2. **Claim:** DQN and its variants (e.g., DDQN, AE‑DQN) are the most frequently used DRL algorithms for NIDS, while distributional RL methods such as QRDQN or C51 have not yet been applied to NIDS in mainstream literature.
    - **Supporting sources:** LopezMartin2020‑DRL‑IDS, AE‑DQN2020, NIDS‑RL‑2023, DQ‑IDS 2025, DRL‑IDS surveys.[^9][^8][^21][^6][^1][^15]
    - **Strength:** Moderate–strong; surveys and key method papers list DQN, DDQN, PPO, A2C, SAC, DDPG but not QRDQN/C51.
    - **Caveat:** There may be unpublished or niche works using distributional RL; state this as “we did not find any clear application of QRDQN to NIDS” rather than “none exist”.
3. **Claim:** RL‑based NIDS have been evaluated primarily on NSL‑KDD and other legacy datasets, with more recent work extending to UNSW‑NB15, CICIDS2017, CICDDoS2019, and CSE‑CIC‑IDS2018, but still almost always in an offline setting.
    - **Supporting sources:** LopezMartin2020‑DRL‑IDS, LopezMartin2021‑RBF‑OfflineRL, Ren2022‑IDRDRL, AE‑DQN2020, RLTechniques2023.[^6][^2][^4][^10][^1]
    - **Strength:** Strong.
    - **Caveat:** New datasets (IoMT, SDN) are emerging; you should mention them for completeness.
4. **Claim:** Evaluation of RL‑based NIDS typically relies on random train/test splits on a single dataset, with little attention to temporal splits, cross‑dataset generalization, or external validation on lab traffic.
    - **Supporting sources:** LopezMartin2020‑DRL‑IDS, LopezMartin2021‑RBF‑OfflineRL, Ren2022‑IDRDRL, AE‑DQN2020, Sanusi2023‑DRLThesis, DRL‑IDS surveys.[^17][^16][^8][^9][^2][^4][^6][^1]
    - **Strength:** Strong; consistent pattern across many papers.
    - **Caveat:** Some works evaluate on more than one public dataset, which is a weak form of external validation; acknowledge this.
5. **Claim:** Detailed reward design (including cost sensitivity and explicit penalties for false negatives) and class‑imbalance handling are under‑reported in RL‑based NIDS papers, even though class imbalance is a central issue in IDS datasets.
    - **Supporting sources:** LopezMartin2021‑RBF‑OfflineRL, Ren2022‑IDRDRL, DDPG‑IDS 2025, DQ‑IDS 2025, DRL‑IDS surveys.[^17][^8][^2][^4][^7][^15]
    - **Strength:** Moderate; some papers mention imbalance, but few give full reward formulas.
    - **Caveat:** For a few recent works (DDPG‑IDS, HCLR‑IDS), the full paper may specify rewards in more detail than accessible snippets.
6. **Claim:** Multi‑dataset DRL‑IDS evaluations that include several public datasets (e.g., NSL‑KDD, UNSW‑NB15, CICIDS2017, CICDDoS2019) exist, but they are still conducted in offline settings without evaluation on independent lab‑captured traffic.
    - **Supporting sources:** LopezMartin2021‑RBF‑OfflineRL (5 datasets), DRL‑IDS surveys, ID‑RDRL 2022.[^8][^9][^2][^4]
    - **Strength:** Strong.
    - **Caveat:** Some cyber‑range works evaluate ML‑NIDS in labs, but not with RL agents.
7. **Claim:** Given the current literature, a QRDQN‑based flow‑level defender trained on CICIDS2017 and evaluated both within‑dataset (with careful splits) and on lab‑captured traffic would constitute a novel and methodologically stronger contribution, especially if reward design reflects cost asymmetry between false positives and false negatives.
    - **Supporting sources:** All of the above plus general ML‑NIDS dataset and evaluation critiques (e.g., Tavallaee2009, Generalizability2025, DL‑IDS surveys).[^33][^37][^38][^35][^9][^17][^2][^4][^8][^10][^1]
    - **Strength:** Moderate–strong; no existing RL‑IDS uses distributional RL with such an evaluation pipeline (as far as the search reveals).
    - **Caveat:** Phrase as “to the best of our knowledge, we did not find…” and emphasise methodological improvements rather than uniqueness.

***

## 8. BibTeX candidates

*(Adapt field names and capitalization to your thesis style. Some DOIs are known; leave DOI blank where uncertain.)*

```bibtex
@article{LopezMartin2020DRLIDS,
  author  = {Lopez-Martin, Manuel and Carro, Bel{\'e}n and Sanchez-Esguevillas, Antonio},
  title   = {Application of Deep Reinforcement Learning to Intrusion Detection for Supervised Problems},
  journal = {Expert Systems with Applications},
  volume  = {141},
  pages   = {112963},
  year    = {2020},
  doi     = {10.1016/j.eswa.2019.112963}
}

@article{LopezMartin2021RBFOfflineRL,
  author  = {Lopez-Martin, Manuel and Sanchez-Esguevillas, Antonio and Arribas, Juan I. and Carro, Bel{\'e}n},
  title   = {Network Intrusion Detection Based on Extended RBF Neural Network With Offline Reinforcement Learning},
  journal = {IEEE Access},
  volume  = {9},
  pages   = {153153--153170},
  year    = {2021},
  doi     = {10.1109/ACCESS.2021.3127689}
}

@article{Ren2022IDRDRL,
  author  = {Ren, Kezhou and others},
  title   = {ID-RDRL: A Deep Reinforcement Learning-Based Feature Selection Intrusion Detection Model},
  journal = {Scientific Reports},
  volume  = {12},
  number  = {15370},
  year    = {2022},
  doi     = {10.1038/s41598-022-19366-3}
}

@article{AEDDQN2020NSLKDD,
  author  = {FirstAuthor, A. and Others}, % fill actual author list
  title   = {Network Intrusion Detection Systems Using Adversarial Reinforcement Learning with Deep Q-Network},
  journal = {IEEE Access},
  year    = {2020},
  note    = {Accessed via IEEE Xplore, document ID 9289884}
}

@inproceedings{NIDSRL2023,
  author    = {FirstAuthor, A. and Others},
  title     = {Network Intrusion Detection System Using Reinforcement Learning},
  booktitle = {2023 IEEE Conference on ...}, % fill venue name
  year      = {2023},
  note      = {Document ID 10170630 on IEEE Xplore}
}

@inproceedings{RLTechniques2023NIDS,
  author    = {FirstAuthor, A. and Others},
  title     = {Network Intrusion Detection System Using Reinforcement Learning Techniques},
  booktitle = {2023 IEEE Conference on ...}, % fill venue
  year      = {2023},
  note      = {Document ID 10245608 on IEEE Xplore}
}

@mastersthesis{Sanusi2023DRLIDS,
  author       = {Sanusi, Hamed T.},
  title        = {Network Intrusion Detection Using Deep Reinforcement Learning},
  school       = {Georgia Southern University},
  year         = {2023},
  address      = {USA},
  url          = {https://digitalcommons.georgiasouthern.edu/etd/2676}
}

@article{Hossain2025DQIDS,
  author  = {Hossain, M. A. and others},
  title   = {Deep Q-Learning Intrusion Detection System (DQ-IDS): A Novel Reinforcement Learning Approach for Adaptive and Self-Learning Cybersecurity},
  journal = {...},  % fill journal name once confirmed
  year    = {2025},
  note    = {Accessed via ScienceDirect / DOAJ}
}

@article{DDPG2025AttackDetection,
  author  = {FirstAuthor, A. and Others},
  title   = {Optimized Network Security Attack Detection Algorithm Based on Deep Deterministic Policy Gradient (DDPG)},
  journal = {IEEE Journal / Conference}, % exact venue from IEEE
  year    = {2025},
  note    = {IEEE Xplore document ID 11189334}
}

@article{DRLIDSSDN2025,
  author  = {Hossain, M. A. and others}, % verify authorship
  title   = {Deep Reinforcement Learning-Based Intrusion Detection Scheme for Software-Defined Networking},
  journal = {Scientific Reports},
  year    = {2025},
  note    = {Nature article ID s41598-025-24869-w}
}

@article{HCLRIDS2025IoMT,
  author  = {FirstAuthor, A. and Others},
  title   = {A Deep Reinforcement Learning-Based Robust Intrusion Detection System for Securing IoMT Healthcare Networks},
  journal = {Frontiers in Medicine},
  year    = {2025},
  note    = {Article 1524286}
}
```

You can refine author lists and venues once you download the full PDFs; the keys here (LopezMartin2020DRLIDS, LopezMartin2021RBFOfflineRL, Ren2022IDRDRL, etc.) are suitable to reference in your text.

***

## 9. Codex handoff

*For drafting the subsection: “Aprendizaje por refuerzo aplicado a sistemas de detección de intrusiones”*

**Objective for Codex:**
Write a Spanish subsection of the State of the Art chapter that explains how RL/DRL has been applied to NIDS and network attack classification, with emphasis on dataset‑as‑environment formulations and on how this relates to a binary PERMIT/BLOCK flow‑based defender using QRDQN.

**Recommended structure (subsection outline):**

1. **Introducción al uso de RL en NIDS**
    - Explicar brevemente por qué el aprendizaje por refuerzo es atractivo para ciberseguridad (capacidad de adaptación, formulación secuencial).
    - Introducir la idea general de “clasificación mediante RL”: el agente observa un vector de características de flujo y decide una etiqueta (normal/ataque).[^2][^4][^1]
2. **Formulación dataset‑as‑environment**
    - Describir la contribución conceptual de López‑Martín: modificación del paradigma clásico de DRL donde el “entorno” es una función de muestreo sobre registros etiquetados.[^5][^1]
    - Explicar que cada muestra o flujo es un estado, la etiqueta predicha es la acción y el entorno devuelve una recompensa basada en la corrección de la clasificación.
    - Destacar que esta formulación es la base de muchos trabajos DRL‑IDS posteriores y que tu entorno Gymnasium sigue esta idea, pero aplicado a flujos de CICIDS2017 con una acción binaria PERMIT/BLOCK.
3. **Algoritmos DRL empleados en la literatura**
    - Resumir que la mayoría de los trabajos se basan en DQN y variantes (DDQN, AE‑DQN), además de enfoques actor‑critic (AC, DDPG) y algunos Policy Gradient.[^9][^21][^8][^6][^1][^7]
    - Señalar que no se han encontrado aplicaciones claras de métodos distribucionales (p.ej. QRDQN, C51) a NIDS; tu trabajo introduce esta familia de algoritmos en este contexto.
4. **Trabajos representativos y datasets utilizados**
    - Describir sintéticamente:
        - López‑Martín 2020 (DRL para NSL‑KDD y AWID; DQN, DDQN, PG, AC).[^5][^1]
        - López‑Martín 2021 (RBFNN con RL offline sobre NSL‑KDD, UNSW‑NB15, AWID, CICIDS2017 y CICDDoS2019).[^13][^4]
        - ID‑RDRL 2022 (DRL con selección de características sobre CSE‑CIC‑IDS2018).[^2]
        - AE‑DQN y extensiones (AE‑DQN, NIDS‑RL) sobre NSL‑KDD.[^21][^6]
        - NIDS con RL en NSL‑KDD y CICIDS2017 mediante DQN multi‑agente (RLTechniques2023).[^10]
        - Trabajos recientes con DQ‑IDS, DDPG‑IDS, IoMT/SDN (sólo como ejemplos breves).[^12][^11][^7][^15]
    - Resaltar qué datasets se usan (NSL‑KDD, UNSW‑NB15, CICIDS2017, CSE‑CIC‑IDS2018, CICDDoS2019).
5. **Diseño de la recompensa y manejo del desequilibrio de clases**
    - Clasificar las funciones de recompensa:
        - Simple (correcto/incorrecto).[^16][^1][^15]
        - Coste‑sensible / multi‑objetivo (p.ej., precisión frente a latencia en DDPG‑IDS).[^11][^7]
        - Intentos de tratar el desequilibrio de clases mediante re‑muestreo o priorización (RBF‑RL, ID‑RDRL, prioridad de experiencias en DDPG).[^4][^2][^7]
    - Indicar que en general el diseño de recompensa está poco detallado y rara vez se calibra explícitamente el coste relativo de falsos positivos y falsos negativos.
6. **Comparación con modelos supervisados clásicos**
    - Explicar que la mayoría de trabajos comparan sus agentes RL/DRL con baselines supervisados (Random Forest, SVM, MLP, CNN/LSTM).[^6][^1][^4][^10][^2]
    - Mencionar que tu tesis seguirá esta práctica incluyendo al menos un baseline de ML clásico (p.ej. Random Forest) y uno profundo (MLP) sobre los mismos vectores de características.
7. **Limitaciones metodológicas de los trabajos RL‑NIDS**
    - Subrayar problemas comunes: divisiones aleatorias en datasets con fuerte estructura temporal, ausencia de validación externa en tráfico real o de laboratorio, poca atención a desequilibrios de clases y coste de errores, ausencia de código y especificación incompleta de la recompensa.[^17][^8][^9][^21][^1][^4][^10][^6][^2]
    - Utilizar estas críticas para motivar decisiones metodológicas de tu prototipo: particiones temporales o por escenarios en CICIDS2017, evaluación offline sobre tráfico capturado en un laboratorio privado, recompensas coste‑sensibles y publicación de detalles de preprocesado y divisiones.
8. **Posicionamiento de tu trabajo**
    - Explicar que tu prototipo extiende la familia de trabajos dataset‑as‑environment a un agente de defensa binario sobre flujos de red, entrenado con QRDQN y evaluado tanto en CICIDS2017 como en tráfico de laboratorio.
    - Destacar que la contribución principal no es únicamente el algoritmo (QRDQN), sino la combinación de: representación basada en flujos, formulación RL explícita PERMIT/BLOCK, diseño de recompensa coste‑sensible y validación externa.

**Style instructions for Codex:**

- Escribir en un tono académico, crítico pero equilibrado, evitando afirmaciones absolutas (“primero”, “único”) salvo cuando haya consenso claro.
- Introducir cada trabajo representativo en 2–3 frases, enfatizando: algoritmo RL, dataset, formulación del entorno, forma de recompensa y debilidades metodológicas.
- Hacer referencias cruzadas a las secciones de datasets y de problemas de evaluación del capítulo (por ejemplo: “como se discute en la Sección 3, CICIDS2017 presenta correlaciones temporales que no se tienen en cuenta en estos trabajos”).
- Mantener el foco en cómo estos trabajos preparan el terreno para un agente PERMIT/BLOCK sobre flujos con QRDQN, dejando los detalles de implementación para el capítulo de metodología/implementación.
<span style="display:none">[^39][^40][^41][^42][^43][^44][^45][^46][^47][^48][^49][^50][^51][^52][^53][^54]</span>

<div align="center">⁂</div>

[^1]: https://dl.acm.org/doi/abs/10.1016/j.eswa.2019.112963

[^2]: https://pmc.ncbi.nlm.nih.gov/articles/PMC9470692/

[^3]: https://www.nature.com/articles/s41598-022-19366-3

[^4]: https://gredos.usal.es/handle/10366/154846?show=full\&locale-attribute=de

[^5]: https://github.com/sophiaabolore/DRL-Anomaly-Based-IDS

[^6]: https://ieeexplore.ieee.org/document/9289884/

[^7]: https://ieeexplore.ieee.org/document/11189334/

[^8]: https://arxiv.org/abs/2405.20038

[^9]: https://onlinelibrary.wiley.com/doi/abs/10.1002/ail2.70026

[^10]: https://ieeexplore.ieee.org/document/10245608/

[^11]: https://www.nature.com/articles/s41598-025-24869-w

[^12]: https://www.frontiersin.org/articles/10.3389/fmed.2025.1524286/full

[^13]: https://uvadoc.uva.es/bitstream/handle/10324/54210/Network-intrusion-detection-based-extended.pdf?sequence=1\&isAllowed=y

[^14]: https://www.academia.edu/122380343/A_novel_reinforcement_learning_based_hybrid_intrusion_detection_system_on_fog_to_cloud_computing

[^15]: https://doaj.org/article/594d225c3e3449dfa815eb2875cc3677

[^16]: https://digitalcommons.georgiasouthern.edu/cgi/viewcontent.cgi?article=3909\&context=etd

[^17]: https://dl.acm.org/doi/10.1016/j.comnet.2023.110016

[^18]: https://dl.acm.org/doi/10.1016/j.eswa.2019.112963

[^19]: https://www.sciencedirect.com/science/article/abs/pii/S0957417419306815

[^20]: https://ui.adsabs.harvard.edu/abs/2021IEEEA...9o3153L/abstract

[^21]: https://ieeexplore.ieee.org/document/10170630/

[^22]: https://www.semanticscholar.org/paper/Deep-Q-learning-intrusion-detection-system-A-novel-Hossain/851c7a9d7efc20dca07c5e2296c6cfcc261496e6

[^23]: https://linkinghub.elsevier.com/retrieve/pii/S2405959525000694

[^24]: https://oipub.com/papers/270166643

[^25]: https://uvadoc.uva.es/handle/10324/54301?locale-attribute=en

[^26]: https://uvadoc.uva.es/bitstream/handle/10324/54301/Adversarial-environment-reinforcement-learning.pdf;jsessionid=AAA40333F7DA6C95BB36336854966176?sequence=1

[^27]: https://www.sciencedirect.com/science/article/abs/pii/S0167404823004121

[^28]: https://www.mdpi.com/1999-4893/18/12/749

[^29]: http://www.jfdc.cnic.cn/EN/10.11871/jfdc.issn.2096-742X.2021.03.006

[^30]: https://www.cs.unb.ca/research-expo/expos/2018/submissions/20180403-14-56-isharafa-at-unb.ca-toward_generating_a_new_intrusion_detection_dataset_and_intrusion_traffic_characterization.pdf

[^31]: https://www.semanticscholar.org/paper/Toward-Generating-a-New-Intrusion-Detection-Dataset-Sharafaldin-Lashkari/a27089efabc5f4abd5ddf2be2a409bff41f31199

[^32]: https://www.sciencedirect.com/science/article/pii/S1877050920307961

[^33]: https://www.mdpi.com/2076-3417/15/15/8466

[^34]: https://github.com/bz-hadjer/master_project

[^35]: https://www.semanticscholar.org/paper/A-detailed-analysis-of-the-KDD-CUP-99-data-set-Tavallaee-Bagheri/fc3eb090e39d71295c362458b8a0c48d2c5d8377

[^36]: https://peerj.com/preprints/1954v1.pdf

[^37]: https://onlinelibrary.wiley.com/doi/10.1155/2023/6048087

[^38]: https://arxiv.org/abs/2504.07839

[^39]: https://dl.acm.org/doi/10.1145/3802927.3802935

[^40]: https://www.sciencedirect.com/science/article/pii/S2405959525000694

[^41]: https://gredos.usal.es/handle/10366/154846?show=full\&locale-attribute=en

[^42]: https://www.sciencedirect.com/science/article/abs/pii/S0167865524002137

[^43]: https://www.facebook.com/ah.cse.dme/posts/alhamdulillah-happy-to-share-my-latest-research-paper-title-deep-q-learning-intr/2730428883813023/

[^44]: https://www.ijsce.org/portfolio-item/F369815060126/

[^45]: https://ieeexplore.ieee.org/document/10986548/

[^46]: https://ieeexplore.ieee.org/document/11253990/

[^47]: https://ieeexplore.ieee.org/document/11049980/

[^48]: https://ieeexplore.ieee.org/document/10495036/

[^49]: https://journals.abuad.edu.ng/index.php/ajerd/article/view/1549

[^50]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12749483/

[^51]: https://discovery.researcher.life/article/application-of-deep-reinforcement-learning-for-intrusion-detection-in-internet-of-things-a-systematic-review/ba20084d3e52319f850044b760aaff6f

[^52]: https://onlinelibrary.wiley.com/doi/full/10.1155/jama/9547540

[^53]: https://www.ijraset.com/research-paper/analysis-of-efficient-intrusion-detection-system

[^54]: https://ouci.dntb.gov.ua/en/works/logzVvjl/

