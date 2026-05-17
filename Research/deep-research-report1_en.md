# Research Dossier for the State of the Art

## Proposed Chapter Structure

The most solid structure for the State of the Art chapter, given your Bachelor's Thesis (TFG) and the type of claims you want to defend before a tribunal, should not be organized "by algorithmic trend", but by a progression from the engineering and security problem to the exact positioning of the prototype. That reduces the risk of overstating the contribution of RL and helps distinguish between established facts, common practices, and claims that are still weak in the literature. NIDS are part of the broader IDPS family; flow-based systems are common because network telemetry based on flow records is standardized and operationally manageable, and because several public reference benchmarks are already published or derived at the flow level.

It is also convenient for the structure to make visible, from the beginning, an uncomfortable but central fact: ML-based NIDS literature frequently reports near-perfect results within the same dataset, but those results usually degrade under domain shifts, cross-dataset validation, or more realistic traffic. That tension between "internal benchmark" and "transportability" is precisely where your experimental design holds the most argumentative value.

A defensible chapter structure in Spanish would be the following:

| Proposed Subsection | Function in the chapter | What it must demonstrate | Suggested base citations |
|---|---|---|---|
| **Context of network intrusion detection** | Introduce IDS/IDPS, NIDS, signature-based detection and anomaly-based detection | That the problem is not born with ML and that ML does not replace the entire defensive stack | [ScarfoneMell2007], [Ahmad2021] |
| **NIDS based on network traffic and flow-based approaches** | Explain why flows are used and what is gained/lost compared to packets or payloads | Scalability, standardization, compatibility with encrypted traffic, lower semantic granularity | [Claise2013], [Sharafaldin2018], [UNB_IDS2018_Page], [Sarhan2020], [ElMahdaouy2026_weak] |
| **Public datasets for NIDS research** | Present CICIDS2017, CSE-CIC-IDS2018, UNSW-NB15 and NSL-KDD | That they are useful benchmarks but do not equate to real deployment | [Sharafaldin2018], [UNB_IDS2018_Page], [MoustafaSlay2015], [Tavallaee2009], [Ring2019], [Goldschmidt2025_weak] |
| **Supervised learning and deep learning for NIDS** | Situate the bulk of the "mainstream" literature | That the reasonable comparative baseline is not just RL, but also strong supervised classifiers | [Ahmad2021], [Maseer2023], [Sarhan2021], [Corea2024] |
| **RL and DRL fundamentals relevant to the TFG** | Introduce MDP formalism, action, reward, value, policy, DQN, PPO, A2C, distributional RL | That QRDQN does not appear "out of nowhere" and derives from a known technical timeline | [SuttonBarto2018], [Mnih2015], [vanHasselt2016], [Wang2016], [Mnih2016], [Schulman2017], [Bellemare2017], [Dabney2018] |
| **RL and DRL applied to NIDS and cyber defense** | Review works that use RL for classification, detection, response, and adaptive defense | That there are precedents, but heterogeneous and methodologically irregular | [Alavizadeh2021], [Strickland2023], [Tellache2024], [Yang2024_weak], [Gueriani2024_weak], [Palmer2023_weak] |
| **Methodological limitations of the field** | Criticize biases, leakage, random splits, lack of external validation, and weak reproducibility | That the chapter does not sell illusory exactitudes | [Arp2020], [Layeghy2021], [Layeghy2022], [Cantone2024], [Ring2019] |
| **Positioning and scope of the TFG** | Close the chapter by connecting with your concrete implementation | That your TFG provides a concrete formulation and a prudent evaluation, not a "definitive solution" | TFG repository + methodological sources |

## Narrative Synthesis

Below, the synthesis is organized as working material so that Codex can later draft academic prose. It is not yet final report text.

| Subsection | Core Idea | Sources to cite | What the thesis should say | What the thesis should avoid saying | Connection with your implementation |
|---|---|---|---|---|---|
| **NIDS Context** | A NIDS monitors network traffic to detect malicious or anomalous activity; it is not synonymous with a firewall nor does it guarantee active prevention | [ScarfoneMell2007], [Ahmad2021] | That NIDS are a monitoring/detection layer within a broader defensive architecture | That "a NIDS blocks attacks" by definition; that corresponds to inline IPS/IDPS | Your prototype makes binary PERMIT/BLOCK decisions, but in the final phase it only performs offline inference, not active blocking. |
| **Flow-based Approach** | Flows aggregate packets into records with statistics; they are useful due to cost, standardization, and availability in routers, collectors, and datasets | [Claise2013], [Sharafaldin2018], [UNB_IDS2018_Page], [Sarhan2020], [ElMahdaouy2026_weak] | That a flow-based approach is reasonable for a TFG because it operates on available and reproducible telemetry, even when the payload is unavailable or encrypted | That flows "capture all the semantics" of the attack; they lose fine content and sequence context | Your pipeline uses a fixed canonical scheme of 76 flow-based features and a missingness mask, producing 152-dimensional observations. |
| **Public Datasets** | Public datasets make it possible to train, compare, and reproduce; at the same time they introduce risks of overfitting to the benchmark | [Ring2019], [Sharafaldin2018], [UNB_IDS2018_Page], [MoustafaSlay2015], [Tavallaee2009], [Goldschmidt2025_weak] | That CICIDS2017, CSE-CIC-IDS2018, UNSW-NB15, and NSL-KDD are widely used references, each with different trade-offs between timeliness, realism, and ease of use | That any of these datasets "represent the real Internet" without reservations | Your TFG uses CICIDS2017 as the main dataset and keeps NSL-KDD only for historical benchmarking, not as a final path towards the laboratory phase. |
| **Supervised and Deep Learning** | Most of the NIDS literature is still, in fact, supervised: trees, RF, SVM, XGBoost, MLP, CNN, RNN/LSTM, hybrids, and combinations with feature selection | [Ahmad2021], [Maseer2023], [Corea2024], [HadiMohammed2022] | That any RL evaluation must be compared at least with one or more strong supervised baselines | That RL automatically displaces supervised learning in this problem | Your report should justify a supervised baseline over the same canonical vector, not just compare between RL variants |
| **RL/DRL Fundamentals** | In RL the agent observes a state, chooses an action, and receives a reward; DQN and its variants are especially relevant when the action is discrete | [SuttonBarto2018], [Mnih2015], [vanHasselt2016], [Wang2016], [Mnih2016], [Schulman2017], [Bellemare2017], [Dabney2018] | That your formulation converts each flow into an observation and each defense decision into a binary action, with asymmetric reward for FP/FN | That this formulation faithfully reproduces all the sequential dynamics of an operational network | Your environment defines `0 = PERMIT` and `1 = BLOCK`, with differentiated rewards for TP, FP, and FN; this is consistent with a discrete decision problem, but sequentially simplified. |
| **RL/DRL for NIDS and Defense** | There are works that reformulate intrusion detection or classification as an RL problem, but a significant portion of them look more like classification with reward than rich sequential control | [Alavizadeh2021], [Strickland2023], [Tellache2024], [Yang2024_weak], [Gueriani2024_weak], [Palmer2023_weak] | That RL in cybersecurity covers at least two families: detection/classification on datasets and adaptive defense/response in more complex environments | That every RL work for NIDS is directly comparable with your prototype | Your project is closer to the "dataset-as-environment, sample-as-state, action-as-label/decision" family than to online autonomous defense over topologies or multi-agent simulators |
| **Methodological Limitations** | The literature suffers from metric inflation, weak validations, and poor generalization between domains | [Arp2020], [Layeghy2021], [Layeghy2022], [Cantone2024], [Ring2019] | That the value of your TFG depends more on methodological prudence than on chasing a maximum accuracy figure | That a high score on a random split would suffice to speak of real applicability | Your repository already distinguishes between random split, hard split by day/CSV, anti-leakage checks, and leave-one-CSV-out validation; that is a strength that should be recounted accurately. |
| **TFG Positioning** | The reasonable contribution is not "inventing RL for NIDS", but studying an explicit binary formulation of an RL defender with QRDQN and more cautious evaluation | [Yang2024_weak], [Gueriani2024_weak], [Alavizadeh2021], [Tellache2024], [Arp2020], [Cantone2024] | That the TFG explores a concrete, reproducible, and bounded configuration: PERMIT/BLOCK per flow, offline training on a public benchmark, and external offline validation on private lab traffic | That the TFG resolves the operational deployment of a real NIDS/IPS | The repository makes it clear that phase 1 is offline training/validation on datasets and phase 2 is offline inference on custom traffic; active blocking is not implemented. |

There is also an important observation for the final drafting: in the reviewed literature, DQN and its derivatives have much more visibility than explicit distributional RL in NIDS; recent surveys on DRL for NIDS emphasize DQN, actor-critic architectures, and hybrids, while pointing out that many recent DRL technologies remain underexplored. Therefore, a prudent framing for QRDQN would be "a technically plausible and underrepresented algorithm in this subfield", not "a confirmed new state-of-the-art in NIDS".

It is also important to separate very carefully the logic of the RL environment from the logic of defensive deployment. In your implementation, the environment is a sequential classification Gym over labeled samples, where the action selects to permit or block, and the reward specifically penalizes false negatives; this is fine for an academic prototype, but the thesis should not present that sequence of samples as a faithful simulation of a network in production.

## Source Base

### Source Matrix

The following matrix prioritizes sources with the greatest argumentative weight: standards, official dataset pages, foundational papers, generalization/evaluation works, and the closest RL precedents. When a source is a preprint or the evidence is still weak, I mark it explicitly. The literature and official pages agree that public datasets remain essential due to the scarcity of shareable real traces, but also that this dependence produces a persistent gap between benchmark and deployment.

| Citation key | Full reference | Year | Type | Topic | Method / algorithm | Dataset(s) | Task | Metrics | Evaluation protocol | Main contribution | Limitations | Relevance to your TFG | Recommended section | DOI / arXiv / link |
|---|---|---:|---|---|---|---|---|---|---|---|---|---|---|---|
| **ScarfoneMell2007** | Scarfone, K. A., Mell, P. M. *Guide to Intrusion Detection and Prevention Systems (IDPS)*, NIST SP 800-94 | 2007 | standard/guide | IDS/IDPS | — | — | Conceptual framework | — | Technical guide | Defines IDPS classes and operational context | Old; does not cover modern ML | High for base framework | NIDS Context | NIST SP 800-94 |
| **Claise2013** | Claise, B., Trammell, B., Aitken, P. *RFC 7011: Specification of the IPFIX Protocol* | 2013 | standard | Flow telemetry | IPFIX | — | Standardization | — | IETF standard | Formal basis for exchange of flow information | Not specific to NIDS | High to justify flow | Flow-based | RFC 7011 |
| **Sharafaldin2018** | Sharafaldin, I., Lashkari, A. H., Ghorbani, A. A. *Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization* | 2018 | dataset paper | CICIDS2017 | CICFlowMeter features | CICIDS2017 | Dataset / benchmark | — | Dataset presentation | Canonical document associated with CICIDS2017 | Synthetic dataset; not equivalent to a production network | Very high | Datasets | ICISSP 2018 |
| **UNB_IDS2017_Page** | UNB CIC. *Intrusion detection evaluation dataset (CIC-IDS2017)* | 2026 acc. | official dataset page | CICIDS2017 | >80 flow features | CICIDS2017 | Dataset / benchmark | — | Official documentation | Dates, attacks, design criteria, features | Institutional page, not peer-reviewed paper | Very high | Datasets | UNB dataset page |
| **UNB_IDS2018_Page** | UNB CIC. *CSE-CIC-IDS2018 on AWS* | 2026 acc. | official dataset page | CSE-CIC-IDS2018 | CICFlowMeter-V3 | CSE-CIC-IDS2018 | Dataset / benchmark | — | Official documentation | Describes motivation, attack lists, and features | Does not replace an independent canonical paper | High | Datasets | UNB dataset page |
| **MoustafaSlay2015** | Moustafa, N., Slay, J. *UNSW-NB15: a comprehensive data set for network intrusion detection systems* | 2015 | dataset paper | UNSW-NB15 | Argus/Bro-derived 49 features | UNSW-NB15 | Dataset / benchmark | — | Published train/test split | Modern dataset compared to KDD, with 9 attack types | Also synthetic/hybrid | Very high | Datasets | UNSW page / MilCIS 2015 |
| **Tavallaee2009** | Tavallaee, M., Bagheri, E., Lu, W., Ghorbani, A. A. *A Detailed Analysis of the KDD CUP 99 Data Set* y base NSL-KDD | 2009 | dataset critique | NSL-KDD | — | KDD99 / NSL-KDD | Benchmark | — | Dataset redesign | Justifies NSL-KDD against KDD99 redundancies | Still unrealistic for modern traffic | High, but with caveat | Datasets | NSL-KDD / paper |
| **Ring2019** | Ring, M. et al. *A Survey of Network-based Intrusion Detection Data Sets* | 2019 | survey | NIDS datasets | — | multiple | Dataset survey | — | Comparative review | Synthesizes properties of NIDS datasets | Preprint in evidence consulted | Very high | Datasets / methodology | arXiv:1903.02460 |
| **Goldschmidt2025_weak** | Goldschmidt, P., Chudá, D. *Network Intrusion Datasets: A Survey, Limitations, and Recommendations* | 2025 | survey, **preprint** | NIDS datasets | — | 89 datasets | SLR | — | SLR | Popularity and recent recommendations | As of review date, evidence located as preprint | Moderate | Datasets / limitations | arXiv:2502.06688 |
| **Ahmad2021** | Ahmad, Z. et al. *Network intrusion detection system: A systematic study of machine learning and deep learning approaches* | 2021 | survey | ML/DL for NIDS | varied ML/DL | multiple | Survey | accuracy, precision, recall, etc. | Systematic review | Widely cited panoramic reference | Not a benchmark itself | Very high | Supervised/DL | TETT 2021 |
| **Maseer2023** | Maseer, Z. K. et al. *Meta-Analysis and Systematic Review for Anomaly NIDS* | 2023 | survey, **preprint** | methods, datasets, validation | ML/DL | multiple | SLR/meta-analysis | various | review | Emphasizes validation and challenges | Evidence located as preprint | Moderate | Supervised/DL / methodology | arXiv:2308.02805 |
| **Sarhan2020** | Sarhan, M. et al. *NetFlow Datasets for Machine Learning-based NIDS* | 2020 | benchmark/dataset | flow-based | NetFlow features | NF-UNSW, NF-CSE-CIC-IDS2018, etc. | dataset + baseline | binary/multiclass performance | Conversion to a common feature set | Advocates for a common and operational feature space | Preliminary results; does not resolve entire domain | Very high | Flow-based / datasets | arXiv:2011.09144 |
| **Sarhan2021** | Sarhan, M., Layeghy, S., Portmann, M. *Evaluating Standard Feature Sets Towards Increased Generalisability and Explainability of ML-based NIDS* | 2021 | benchmark/method | generalization | ML + SHAP | CSE-CIC-IDS2018, BoT-IoT, ToN-IoT | Binary/multiclass | accuracy, explainability | Cross-dataset feature comparison | Shows utility of standardized feature sets | Does not study RL | High | Supervised / methodology | arXiv:2104.07183 |
| **Layeghy2021** | Layeghy, S., Gallagher, M., Portmann, M. *Benchmarking the Benchmark: Analysis of Synthetic NIDS Datasets* | 2021 | benchmark critique | dataset realism | statistical analysis | various synthetic + real networks | dataset comparison | statistical distributions | Comparison synthetic vs real | Questions transfer to production | Not an evaluation of RL models | Very high | Methodological limitations | arXiv:2104.09029 |
| **Layeghy2022** | Layeghy, S., Portmann, M. *On Generalisability of ML-based NIDS* | 2022 | benchmark critique | generalization | 7 sup./unsup. models | 4 datasets | cross-dataset NIDS | various | train on one domain, test on another | Shows poor generalization and asymmetry between domains | Does not cover RL | Very high | Methodological limitations | arXiv:2205.04112 |
| **Cantone2024** | Cantone, M., Marrocco, C., Bria, A. *On the Cross-Dataset Generalization of Machine Learning for NIDS* | 2024 | benchmark | cross-dataset | 4 classifiers | CIC-IDS2017, CSE-CIC-IDS2018, LycoS... | generalization | accuracy, etc. | cross-dataset | Almost random performance across datasets in many cases | Preprint in evidence reviewed | Very high | Limitations / external validation | arXiv:2402.10974 |
| **Arp2020** | Arp, D. et al. *Dos and Don'ts of Machine Learning in Computer Security* | 2020 | methodology | ML in security | — | security in general | critical guide | — | study of 30 papers + empirical analysis | Identifies pitfalls generalizable to cybersecurity | Not specific to NIDS | Very high | Methodology | arXiv:2010.09470 |
| **SuttonBarto2018** | Sutton, R. S., Barto, A. G. *Reinforcement Learning: An Introduction* | 2018 | foundational book | RL | general RL | — | theory | — | — | Conceptual basis of state, action, reward, policy | Not specific to NIDS | Very high | RL Fundamentals | MIT Press |
| **Mnih2015** | Mnih, V. et al. *Human-level Control through Deep Reinforcement Learning* | 2015 | method paper | DQN | DQN | Atari | control | score | benchmark RL | DQN with replay and target network | Not cybersecurity | Very high | RL Fundamentals | Nature / doi:10.1038/nature14236 |
| **vanHasselt2016** | van Hasselt, H., Guez, A., Silver, D. *Deep Reinforcement Learning with Double Q-learning* | 2016 | method paper | Double DQN | DDQN | Atari | control | score | benchmark RL | Reduces overestimation | Not NIDS | High | RL Fundamentals | AAAI |
| **Wang2016** | Wang, Z. et al. *Dueling Network Architectures for Deep RL* | 2016 | method paper | Dueling DQN | Dueling DQN | Atari | control | score | benchmark RL | Separates state value and advantage | Not NIDS | High | RL Fundamentals | ICML/PMLR |
| **Mnih2016** | Mnih, V. et al. *Asynchronous Methods for Deep RL* | 2016 | method paper | A3C/A2C family | actor-critic | Atari et al. | control | score | benchmark RL | Basis of A3C; A2C derives as practical synchronous variant | Not NIDS | High | RL Fundamentals | ICML |
| **Schulman2017** | Schulman, J. et al. *Proximal Policy Optimization Algorithms* | 2017 | method paper | PPO | PPO | multiple | control | reward | benchmark RL | Robust and popular actor-critic algorithm | Not NIDS | High | RL Fundamentals | arXiv:1707.06347 |
| **Bellemare2017** | Bellemare, M. G. et al. *A Distributional Perspective on Reinforcement Learning* | 2017 | method paper | distributional RL | C51 | Atari | control | score | benchmark RL | Introduces distributional RL | Not NIDS | Very high | RL Fundamentals | ICML/PMLR |
| **Dabney2018** | Dabney, W. et al. *Distributional Reinforcement Learning with Quantile Regression* | 2018 | method paper | QR-DQN | QRDQN | Atari | control | score | benchmark RL | Primary basis of the algorithm used in your TFG | Not NIDS | Very high | RL Fundamentals / connection to TFG | AAAI |
| **Alavizadeh2021** | Alavizadeh, H., Jang-Jaccard, J., Alavizadeh, H. *Deep Q-Learning based RL Approach for Network Intrusion Detection* | 2021 | primary paper | RL for NIDS | DQL | NSL-KDD | detection/classification | accuracy and classes | episodic training | Direct precedent of RL-classification for IDS | Old dataset; limited realism | Very high | RL for NIDS | arXiv:2111.13978 |
| **Strickland2023** | Strickland, C. et al. *DRL-GAN: A Hybrid Approach for Binary and Multiclass Network Intrusion Detection* | 2023 | primary paper | RL + synthetic data | DRL + GAN | NSL-KDD | binary/multiclass IDS | classification | comparison real/synthetic dataset | Shows use of RL as an enhanced classifier | Mixes two techniques; old dataset; preprint evidence | High | RL for NIDS | arXiv:2301.03368 |
| **Tellache2024** | Tellache, A. et al. *Multi-agent Reinforcement Learning-based Network Intrusion Detection System* | 2024 | primary paper, **preprint** | RL for NIDS | enhanced multi-agent DQN | CIC-IDS2017 | fine detection and classification | FPR, detection rate | CICIDS2017 | RL closer to modern dataset | Evidence located as preprint; methodology audit needed | High | RL for NIDS | arXiv:2407.05766 |
| **Yang2024_weak** | Yang, W. et al. *A Survey for Deep Reinforcement Learning Based Network Intrusion Detection* | 2024 | survey, **preprint** | RL for NIDS | DQN, actor-critic, hybrids | various | survey | various | review | Points out concentration in certain families and gaps | Preprint | Moderate | RL for NIDS / gap | arXiv:2410.07612 |
| **Gueriani2024_weak** | Gueriani, A. et al. *Deep Reinforcement Learning for Intrusion Detection in IoT: A Survey* | 2024 | survey, **preprint** | DRL IDS IoT | various DRL families | various | survey | various | review | Summarizes datasets, metrics, and categories | IoT-centered; preprint | Moderate | RL for NIDS | arXiv:2405.20038 |
| **Palmer2023_weak** | Palmer, G. et al. *Deep Reinforcement Learning for Autonomous Cyber Defence: A Survey* | 2023 | survey, **preprint** | autonomous defense | general DRL | ACD environments | survey | — | review | Differentiates detection on datasets from realistic autonomous defense | Not specific to per-flow classification | High | RL and cyber defense | arXiv:2310.07745 |
| **She2025_weak** | She, Y. *A Robust PPO-optimized Tabular Transformer Framework for Intrusion Detection in IIoT Systems* | 2025 | primary paper, **preprint** | RL + PPO as classifier | PPO + TabTransformer | TON_IoT | IDS | macro-F1, accuracy | evaluation on benchmark | Points out recent precedents of PPO for IDS classification | Weak evidence; only preprint in reviewed sample | Moderate-low | RL for NIDS / algorithms | arXiv:2505.18234 |

### Essential Selection

If you had to reduce the bibliography of the chapter to a very defensible core, this would be the essential selection.

**NIDS and flow-based approaches**: [ScarfoneMell2007], [Claise2013], [Sarhan2020], [Sarhan2021], [ElMahdaouy2026_weak].

**Public datasets**: [Sharafaldin2018], [UNB_IDS2017_Page], [UNB_IDS2018_Page], [MoustafaSlay2015], [Tavallaee2009], [Ring2019], [Goldschmidt2025_weak].

**Supervised ML/DL for NIDS**: [Ahmad2021], [Maseer2023], [Sarhan2021], [Corea2024].

**RL/DRL Fundamentals**: [SuttonBarto2018], [Mnih2015], [vanHasselt2016], [Wang2016], [Mnih2016], [Schulman2017], [Bellemare2017], [Dabney2018]. These are stable foundational sources and do not require recent updates to be valid.

**RL/DRL for NIDS**: [Alavizadeh2021], [Strickland2023], [Tellache2024], [Yang2024_weak], [Gueriani2024_weak].

**Cyber defense and adaptive defense**: [Palmer2023_weak], and as an offensive/adversarial contrast [DeepPackGen2023].

**Methodological risks**: [Arp2020], [Layeghy2021], [Layeghy2022], [Cantone2024].

## Near State of the Art and Methodological Critique

### Works closer to your project

The closest works are not, curiously, those of "autonomous cyber defence" with multiple agents or rich topologies, but those that convert labeled instances of traffic into states and use discrete actions as classes or response decisions. That subgroup is conceptually the direct antecedent of your TFG.

| Paper | What it does | Closeness to your project | Similarities | Differences | Methodological strength | Methodological weakness | Cite as direct precedent? |
|---|---|---|---|---|---|---|---|
| **Alavizadeh2021** | Uses Deep Q-Learning for NIDS over NSL-KDD | **Very high** | RL as classifier; discrete actions; training by rewards; labeled dataset | Old dataset; does not use CICIDS2017; does not use distributional RL | Clear precedent of "sample-as-state, action-as-class/decision" | Risk of over-interpreting RL where the problem is very similar to reinforced supervised classification | **Yes, essential** |
| **Strickland2023** | Combines GAN and DRL for binary and multiclass detection in NSL-KDD | **High** | RL applied to detection/classification over tabular dataset | Adds synthetic generation; hybrid approach | Interesting for minority attacks | Old dataset; more complex and less cleanly comparable | **Yes, as a close precedent but not isomorphic** |
| **Tellache2024** | Multi-agent IDS with enhanced DQN and cost-sensitive learning in CICIDS2017 | **High** | Uses CICIDS2017; RL for detection focusing on imbalance and FPR | Multi-agent architecture, not simple binary; more ambitious objective | Closer in modern dataset | As of review date, located as preprint; demands critical reading of splits and leakage | **Yes** |
| **She2025_weak** | Uses PPO to optimize IDS classification decisions on a table | **Medium** | Shows that PPO is also being used as a classification decision mechanism | IIoT; TabTransformer; another benchmark | Useful for discussion of actor-critic algorithms in IDS | Weak and very recent evidence; shouldn't be over-weighted | **Yes, but only as complementary evidence** |
| **Palmer2023_weak** | Survey of autonomous defense with DRL | **Medium conceptual** | Frames RL beyond flow classification | Focused on full ACD, not per-flow binary decisions | Very useful to delimit scope | Does not serve as a direct benchmark of your prototype | **Yes, to define what your TFG doesn't intend to do** |
| **Yang2024_weak** y **Gueriani2024_weak** | Recent surveys on DRL for NIDS/IoT IDS | **High for context, not for quantitative comparison** | Identify recurring algorithmic families | Surveys, not comparable custom experiments | Good for mapping the field | Preprints; you must note them as such | **Yes** |

The useful conclusion for your chapter is this: the most direct precedent of your approach is not "RL for cyber defence" in the abstract, but the reformulation of IDS classification as an RL decision over labeled samples. This allows you to say that your problem **does have precedents**, but also that these precedents frequently mix the semantics of RL and those of supervised classification, which demands interpretive caution.

### Methodological Critique of the Area

The first recurring weakness is the **intra-dataset performance inflation**. There are multiple works reporting near-perfect precision or F1 within the same benchmark, particularly with synthetic or semi-synthetic datasets, but the critical literature points out that these results have translated poorly to more realistic environments. Layeghy and colleagues show clear statistical differences between synthetic NIDS datasets and real traffic, and both their works and Cantone et al.'s show that cross-dataset performance can drop drastically, sometimes to near-random levels. Arp et al., from a broader ML-in-security perspective, argue that this pattern is not an isolated accident but a symptom of broader methodological pitfalls.

The second weakness is the **excessive use of random splits**. In tabular NIDS, a random split can mix flows from the same scenario, same day, same traffic generator, or same campaign between training and testing. That does not always imply formal leakage in the classical sense, but it can create an overly easy test. Your own repository already reflects this concern by differentiating between random split, hard split by day/CSV, shuffled-label anti-leakage test, and leave-one-exact-CSV-out, which is exactly the kind of discipline the thesis should highlight.

The third weakness is **leakage or use of label proxies**. In flow-based datasets it is very easy for identifiers, ports, absolute timestamps, or export artifacts to act as spurious shortcuts. The best thing about your current implementation is that it formalizes an explicit anti-leakage policy: it excludes IPs, absolute timestamps, Flow IDs, and port fields when they act as label proxies. This connects very well with the general methodological critique of Arp et al. and with the prudent practice required in a serious TFG.

The fourth weakness is the **neglect of imbalance and the asymmetric cost of errors**. In NIDS, a false negative and a false positive do not have the same operational cost. Part of the RL literature precisely justifies the use of asymmetric reward functions to reflect that cost imbalance, but that same flexibility can make comparison across papers opaque if the rewards are not rigorously documented. Here your report should be particularly cautious because the repository contains an informational tension regarding the exact value of the FP penalty between documentation and code; therefore, the thesis should anchor any numerical claim to the exact version of the experiment or artifact used, not to a generic description of the repository.

The fifth weakness is the **absence of external validation**. Many papers train and evaluate on the same public dataset, sometimes even with multiple transformations of the same source. Given that recent literature strongly questions cross-dataset transportability, an additional evaluation over custom laboratory traffic, even if offline and without active blocking, is valuable as proof of robustness under domain shift. But that external validation must be presented for what it is: a **transportability stress test** and not a definitive demonstration of universal operational validity.

The sixth weakness is the **lack of fine reproducibility**: unique seeds, unversioned preprocessing, little clarity on splits, and lack of artifacts. On this point, your repository is better oriented than a good part of the literature, because it already articulates execution identifiers and artifact persistence under `runs/`, in addition to specific validation scripts; however, for the thesis to be fully defensible, it is convenient to explicitly add multiple seeds, data efficiency curves, and supervised baselines under the same preprocessing.

### How your experimental design can answer

The cleanest justification for your design would be the following:

**Internal benchmark on CICIDS2017.** It is reasonable because CICIDS2017 remains a widely used public benchmark, with detailed official documentation, a variety of attacks, and directly usable flow-based CSVs. It serves for reproducibility, internal comparison between variants, and pipeline debugging.

**External validation on private lab traffic.** It is reasonable because the critical literature questions whether intra-dataset performance is transportable. Testing offline inference on flows derived from private captures adds evidence on robustness to distribution shifts, although it does not solely resolve the problem of real deployment.

**Data efficiency curve.** It is useful because it shows whether the RL approach requires disproportionate data volumes compared to supervised baselines. In a TFG this can be more informative than merely seeking the maximum final score.

**Supervised baseline.** It is essential because the problem, as you formulate it, can also be understood as tabular binary classification with asymmetric cost. Without a strong baseline, a tribunal could interpret that RL was chosen because of perceived novelty, not based on comparative evidence.

**Multiple seeds.** They must be used to avoid a single lucky execution dominating the conclusion.

**Error analysis.** Instead of focusing only on global accuracy, it is convenient to analyze false positives and false negatives by traffic families, days, or files, because the cost of these errors is different in defense.

**Fallback strict split.** If time doesn't allow for extensive external validation, the methodological minimum should include at least a hard split by day/CSV and, if possible, leave-one-exact-CSV-out, which your code already contemplates.

### The gap your TFG can claim without exaggerating

The reasonable gap is not "RL is missing in cybersecurity" nor "no one has used RL for IDS". That would be unsustainable. The defensible gap is more concrete:

1. **The literature close to NIDS with RL is dominated by reformulations based on DQN or other generic DRL approaches, with few clear indications of the adoption of distributional RL like QRDQN in this specific subfield.** The evidence here is moderate, not strong, because it relies on recent surveys and the reviewed sample, not on an exhaustive bibliometric mapping.

2. **There is a scarcity of works that, in addition to proposing an RL agent for binary defense decisions over flows, document an explicitly prudent evaluation against leakage, hard splits, and external validation outside the public training dataset.** This does tie in well with the recent methodological critique.

3. **Your TFG does not provide a new RL theory nor a deployed IPS system, but a concrete and reproducible experimental formulation: per-flow binary defender, FP/FN conscious reward engineering, QRDQN in a dataset-as-environment setting, internal benchmark in a public dataset, and offline external testing in custom lab.** That is modest, truthful, and defensible.

## Safe Claims, Claims to Avoid, and Glossary

### Safe Claims

| Claim | Citation keys | Strength of evidence | Caveat | Suggested formulation |
|---|---|---|---|---|
| NIDS are a core monitoring/detection technology, distinct from a traditional firewall and an inline IPS | [ScarfoneMell2007] | **Strong** | Classic guide, not ML-specific | "Network intrusion detection systems constitute a differentiated monitoring and detection layer from traditional filtering mechanisms." |
| Flow-based approaches are common because they rely on standardized and operationally extractable telemetry | [Claise2013], [Sarhan2020], [UNB_IDS2017_Page], [UNB_IDS2018_Page] | **Strong-moderate** | Not all NIDS operate only at the flow level | "Flow-based approaches are particularly attractive when prioritizing scalability, interoperability, and telemetry availability." |
| CICIDS2017, CSE-CIC-IDS2018, UNSW-NB15, and NSL-KDD continue to play an important role as public benchmarks | [Ring2019], [UNB_IDS2017_Page], [UNB_IDS2018_Page], [MoustafaSlay2015], [Tavallaee2009] | **Strong-moderate** | Their popularity does not imply full realism | "These datasets still play a relevant role as experimental baselines, although they present known limitations." |
| In NIDS, the dominant literature is still supervised or supervised DL | [Ahmad2021], [Maseer2023] | **Moderate** | Some reviews found as preprints | "Most of the recent literature in NIDS continues to formulate the problem as supervised classification or supervised deep learning." |
| There are works that formulate intrusion detection as an RL problem on labeled samples | [Alavizadeh2021], [Strickland2023], [Tellache2024] | **Moderate** | Methodological heterogeneity and varying levels of maturity | "There are precedents where intrusion detection or classification is reformulated as a reinforcement decision problem on labeled instances." |
| In the reviewed sample, DQN and related variants predominate; QRDQN does not appear as a widely established option in NIDS | [Yang2024_weak], [Gueriani2024_weak], [Alavizadeh2021], [Tellache2024] | **Moderate-low** | Based on recent surveys and the reviewed sample, not exhaustive bibliometric study | "In the reviewed literature, formulations based on DQN and related architectures predominate, while the use of distributional RL such as QRDQN seems still underrepresented." |
| Near-perfect intra-dataset results do not guarantee generalization outside the benchmark domain | [Layeghy2021], [Layeghy2022], [Cantone2024], [Arp2020] | **Strong** | The exact magnitude of the drop depends on the dataset pair and protocol | "The results obtained within a single benchmark shouldn't be automatically interpreted as evidence of transferability to other environments." |
| Offline external validation on lab traffic is methodologically valuable, even if it does not equate to real deployment | [Layeghy2022], [Cantone2024], [Arp2020] | **Moderate-strong** | It is still private lab, not production | "The additional evaluation on captured lab traffic provides evidence of robustness against distribution change, even without constituting a full operational validation." |
| Your prototype implements a dataset-as-environment with PERMIT/BLOCK binary actions and QRDQN, but no active real-time blocking | TFG repository | **Strong** | The thesis should fix exact version/artifact | "The developed prototype is formulated as a binary decision environment over labeled flows and is evaluated offline; it does not yet implement active inline prevention." |

### Claims to Avoid

| Claim to avoid | Why it is indefensible | Safer alternative |
|---|---|---|
| "This TFG is the first to apply RL to intrusion detection." | False: there are clear precedents | "This TFG explores a concrete and bounded formulation of RL for binary PERMIT/BLOCK decisions on flows." |
| "QRDQN is the standard or dominant algorithm in the area." | There is no solid evidence for that | "QRDQN belongs to the distributional RL family and, in the reviewed sample, appears as an underrepresented option in NIDS." |
| "The results on CICIDS2017 demonstrate effectiveness in real environments." | Critical literature questions that extrapolation | "The results on CICIDS2017 show behavior on a widely used benchmark, but require additional validation outside the dataset domain." |
| "The lab phase validates the system in real conditions." | Your phase 2 is offline inference, not inline deployment | "The lab phase provides offline external validation under custom traffic, useful as proof of partial transportability." |
| "The RL environment faithfully models a real network defense." | The environment is a simplification by samples/rows | "The RL environment constitutes an experimental abstraction of the per-flow decision problem." |
| "RL inherently outperforms supervised learning in NIDS." | There is no general support for such a claim | "The interest in RL here lies in its decision formulation under asymmetric reward; its empirical advantage must be demonstrated against supervised baselines." |
| "Validation on a single random split is sufficient." | Weak against leakage and scenario correlation | "Evaluation should be complemented with hard splits, multiple seeds, and, if possible, external validation." |

### Glossary of Terms

| Term | Definition |
|---|---|
| **NIDS** | Network intrusion detection system that analyzes network traffic or telemetry to identify malicious or anomalous activity. |
| **network flow** | Aggregated record of a network communication between two endpoints during an interval, typically summarized using counters and statistics. |
| **flow features** | Variables derived from the flow, such as duration, bytes, packets, rates, or temporal stats, used as model input. |
| **CICIDS2017** | Public dataset from CIC/UNB with benign traffic and various attacks captured in July 2017, plus PCAPs and flow-level labeled CSVs. |
| **supervised learning** | Learning paradigm where the model is trained with labeled examples to predict a target class or value. |
| **reinforcement learning** | Paradigm where an agent observes a state, chooses an action, and learns to maximize cumulative reward. |
| **state / observation** | Information the agent receives at a given time to decide; in your TFG, the feature vector of a flow. |
| **action** | Decision taken by the agent; in your case, `0 = PERMIT`, `1 = BLOCK`. |
| **reward** | Numerical signal evaluating the quality of the chosen action. |
| **policy** | Rule or function mapping actions to observations. |
| **DQN** | Deep Q-Network, a value-based algorithm for discrete actions that approximates the Q-function with a neural network. |
| **QRDQN** | Distributional variant of DQN based on quantile regression, modeling a distribution of returns instead of just their expectation. |
| **false positive** | Benign instance incorrectly classified as an attack or improperly blocked. |
| **false negative** | Malicious instance incorrectly classified as benign or improperly permitted. |
| **data leakage** | Accidental ingestion of information directly or indirectly correlated with the label in training/evaluation, inflating performance. |
| **external validation** | Evaluation on data distinct from the main training benchmark, ideally from another origin or domain. |
| **distribution shift** | Shift between the distribution of the training data and that of evaluation or deployment. |

## Operational References for Drafting

### Essential BibTeX Block

The following entries are **minimal and usable**. In several cases, it is advisable that, before the final delivery, you normalize them in Zotero/JabRef with exact pages, DOI, and venue when appropriate. I don't include doubtful DOIs; when I don't have them with enough confidence, I leave `url` or `eprint`.

```bibtex
@techreport{ScarfoneMell2007,
  author = {Scarfone, Karen A. and Mell, Peter M.},
  title = {Guide to Intrusion Detection and Prevention Systems (IDPS)},
  institution = {National Institute of Standards and Technology},
  year = {2007},
  number = {NIST SP 800-94},
  url = {https://csrc.nist.gov/pubs/sp/800/94/final}
}

@misc{Claise2013,
  author = {Claise, Benoit and Trammell, Brian and Aitken, Paul},
  title = {RFC 7011: Specification of the IP Flow Information Export (IPFIX) Protocol for the Exchange of Flow Information},
  year = {2013},
  howpublished = {RFC 7011},
  url = {https://www.rfc-editor.org/rfc/rfc7011}
}

@inproceedings{Sharafaldin2018,
  author = {Sharafaldin, Iman and Habibi Lashkari, Arash and Ghorbani, Ali A.},
  title = {Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization},
  booktitle = {Proceedings of the International Conference on Information Systems Security and Privacy},
  year = {2018},
  url = {https://www.unb.ca/cic/datasets/ids-2017.html}
}

@misc{UNB_IDS2017_Page,
  author = {{Canadian Institute for Cybersecurity, UNB}},
  title = {Intrusion Detection Evaluation Dataset (CIC-IDS2017)},
  year = {n.d.},
  url = {https://www.unb.ca/cic/datasets/ids-2017.html},
  note = {Accessed 2026-05-14}
}

@misc{UNB_IDS2018_Page,
  author = {{Canadian Institute for Cybersecurity, UNB}},
  title = {CSE-CIC-IDS2018 on AWS},
  year = {n.d.},
  url = {https://www.unb.ca/cic/datasets/ids-2018.html},
  note = {Accessed 2026-05-14}
}

@inproceedings{MoustafaSlay2015,
  author = {Moustafa, Nour and Slay, Jill},
  title = {UNSW-NB15: A Comprehensive Data Set for Network Intrusion Detection Systems},
  booktitle = {Military Communications and Information Systems Conference},
  year = {2015},
  url = {https://research.unsw.edu.au/projects/unsw-nb15-dataset}
}

@misc{Tavallaee2009,
  author = {Tavallaee, Mahbod and Bagheri, Ebrahim and Lu, Wei and Ghorbani, Ali A.},
  title = {A Detailed Analysis of the KDD CUP 99 Data Set},
  year = {2009},
  note = {Basis for NSL-KDD discussion},
  url = {https://www.unb.ca/cic/datasets/nsl.html}
}

@article{Ring2019,
  author = {Ring, Markus and Wunderlich, Sarah and Scheuring, Deniz and Landes, Dieter and Hotho, Andreas},
  title = {A Survey of Network-based Intrusion Detection Data Sets},
  year = {2019},
  eprint = {1903.02460},
  archivePrefix = {arXiv},
  primaryClass = {cs.CR}
}

@article{Ahmad2021,
  author = {Ahmad, Zeeshan and Khan, Adnan Shahid and Shiang, Cheah Wai and Abdullah, Johari and Ahmad, Farhan},
  title = {Network Intrusion Detection System: A Systematic Study of Machine Learning and Deep Learning Approaches},
  journal = {Transactions on Emerging Telecommunications Technologies},
  year = {2021}
}

@article{Sarhan2020,
  author = {Sarhan, Mohanad and Layeghy, Siamak and Moustafa, Nour and Portmann, Marius},
  title = {NetFlow Datasets for Machine Learning-based Network Intrusion Detection Systems},
  year = {2020},
  eprint = {2011.09144},
  archivePrefix = {arXiv},
  primaryClass = {cs.CR}
}

@article{Sarhan2021,
  author = {Sarhan, Mohanad and Layeghy, Siamak and Portmann, Marius},
  title = {Evaluating Standard Feature Sets Towards Increased Generalisability and Explainability of ML-based Network Intrusion Detection},
  year = {2021},
  eprint = {2104.07183},
  archivePrefix = {arXiv},
  primaryClass = {cs.CR}
}

@article{Layeghy2021,
  author = {Layeghy, Siamak and Gallagher, Marcus and Portmann, Marius},
  title = {Benchmarking the Benchmark: Analysis of Synthetic NIDS Datasets},
  year = {2021},
  eprint = {2104.09029},
  archivePrefix = {arXiv},
  primaryClass = {cs.CR}
}

@article{Layeghy2022,
  author = {Layeghy, Siamak and Portmann, Marius},
  title = {On Generalisability of Machine Learning-based Network Intrusion Detection Systems},
  year = {2022},
  eprint = {2205.04112},
  archivePrefix = {arXiv},
  primaryClass = {cs.CR}
}

@article{Cantone2024,
  author = {Cantone, Marco and Marrocco, Claudio and Bria, Alessandro},
  title = {On the Cross-Dataset Generalization of Machine Learning for Network Intrusion Detection},
  year = {2024},
  eprint = {2402.10974},
  archivePrefix = {arXiv},
  primaryClass = {cs.CR}
}

@article{Arp2020,
  author = {Arp, Daniel and Quiring, Erwin and Pendlebury, Feargus and Warnecke, Alexander and Pierazzi, Fabio and Wressnegger, Christian and Cavallaro, Lorenzo and Rieck, Konrad},
  title = {Dos and Don'ts of Machine Learning in Computer Security},
  year = {2020},
  eprint = {2010.09470},
  archivePrefix = {arXiv},
  primaryClass = {cs.CR}
}

@book{SuttonBarto2018,
  author = {Sutton, Richard S. and Barto, Andrew G.},
  title = {Reinforcement Learning: An Introduction},
  edition = {2},
  publisher = {MIT Press},
  year = {2018}
}

@article{Mnih2015,
  author = {Mnih, Volodymyr and Kavukcuoglu, Koray and Silver, David and Rusu, Andrei A. and others},
  title = {Human-level Control through Deep Reinforcement Learning},
  journal = {Nature},
  volume = {518},
  number = {7540},
  pages = {529--533},
  year = {2015},
  doi = {10.1038/nature14236}
}

@inproceedings{vanHasselt2016,
  author = {van Hasselt, Hado and Guez, Arthur and Silver, David},
  title = {Deep Reinforcement Learning with Double Q-learning},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year = {2016}
}

@inproceedings{Wang2016,
  author = {Wang, Ziyu and Schaul, Tom and Hessel, Matteo and Hasselt, Hado van and Lanctot, Marc and de Freitas, Nando},
  title = {Dueling Network Architectures for Deep Reinforcement Learning},
  booktitle = {Proceedings of the 33rd International Conference on Machine Learning},
  year = {2016}
}

@inproceedings{Mnih2016,
  author = {Mnih, Volodymyr and Badia, Adri\`a Puigdom\`enech and Mirza, Mehdi and Graves, Alex and Lillicrap, Timothy and Harley, Tim and Silver, David and Kavukcuoglu, Koray},
  title = {Asynchronous Methods for Deep Reinforcement Learning},
  booktitle = {Proceedings of the 33rd International Conference on Machine Learning},
  year = {2016}
}

@article{Schulman2017,
  author = {Schulman, John and Wolski, Filip and Dhariwal, Prafulla and Radford, Alec and Klimov, Oleg},
  title = {Proximal Policy Optimization Algorithms},
  year = {2017},
  eprint = {1707.06347},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG}
}

@inproceedings{Bellemare2017,
  author = {Bellemare, Marc G. and Dabney, Will and Munos, R{\'e}mi},
  title = {A Distributional Perspective on Reinforcement Learning},
  booktitle = {Proceedings of the 34th International Conference on Machine Learning},
  year = {2017}
}

@inproceedings{Dabney2018,
  author = {Dabney, Will and Rowland, Mark and Bellemare, Marc G. and Munos, R{\'e}mi},
  title = {Distributional Reinforcement Learning with Quantile Regression},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year = {2018}
}

@article{Alavizadeh2021,
  author = {Alavizadeh, Hooman and Jang-Jaccard, Julian and Alavizadeh, Hootan},
  title = {Deep Q-Learning based Reinforcement Learning Approach for Network Intrusion Detection},
  year = {2021},
  eprint = {2111.13978},
  archivePrefix = {arXiv},
  primaryClass = {cs.CR}
}

@article{Strickland2023,
  author = {Strickland, Caroline and Saha, Chandrika and Zakar, Muhammad and Nejad, Sareh and Tasnim, Noshin and Lizotte, Daniel and Haque, Anwar},
  title = {DRL-GAN: A Hybrid Approach for Binary and Multiclass Network Intrusion Detection},
  year = {2023},
  eprint = {2301.03368},
  archivePrefix = {arXiv},
  primaryClass = {cs.CR}
}

@article{Tellache2024,
  author = {Tellache, Amine and Mokhtari, Amdjed and Korba, Abdelaziz Amara and Ghamri-Doudane, Yacine},
  title = {Multi-agent Reinforcement Learning-based Network Intrusion Detection System},
  year = {2024},
  eprint = {2407.05766},
  archivePrefix = {arXiv},
  primaryClass = {cs.CR}
}

@article{Yang2024_weak,
  author = {Yang, Wanrong and Acuto, Alberto and Zhou, Yihang and Wojtczak, Dominik},
  title = {A Survey for Deep Reinforcement Learning Based Network Intrusion Detection},
  year = {2024},
  eprint = {2410.07612},
  archivePrefix = {arXiv},
  primaryClass = {cs.CR}
}

@article{Gueriani2024_weak,
  author = {Gueriani, Afrah and Kheddar, Hamza and Mazari, Ahmed Cherif},
  title = {Deep Reinforcement Learning for Intrusion Detection in IoT: A Survey},
  year = {2024},
  eprint = {2405.20038},
  archivePrefix = {arXiv},
  primaryClass = {cs.CR}
}

@article{Palmer2023_weak,
  author = {Palmer, Gregory and Parry, Chris and Harrold, Daniel J. B. and Willis, Chris},
  title = {Deep Reinforcement Learning for Autonomous Cyber Defence: A Survey},
  year = {2023},
  eprint = {2310.07745},
  archivePrefix = {arXiv},
  primaryClass = {cs.CR}
}
```

### Precise Handoff for Codex

Draft the **State of the Art / Background** chapter of a Computer Engineering Bachelor's Thesis, in **academic Spanish**, in continuous and well-cohesive prose, **not** as notes or lists. The theme of the thesis is: **"Reinforcement Learning for cybersecurity: an RL-based network-flow defender for binary PERMIT/BLOCK decisions."**

You must create the following subsections in this logical order:

**Context of network intrusion detection.**
Use mainly: [ScarfoneMell2007], [Ahmad2021].
Explain what a NIDS is, how it differs from an IPS/IDPS, and why anomaly detection motivated the use of ML. Do not say that ML replaces all classical approaches.

**Flow-based network traffic NIDS.**
Use: [Claise2013], [Sharafaldin2018], [UNB_IDS2017_Page], [UNB_IDS2018_Page], [Sarhan2020], [Sarhan2021].
Explain what a network flow is, what kind of features are extracted, and why flow-based approaches are frequent. Mention practical advantages and limits: scalability and interoperability versus lower semantic granularity.

**Public datasets in NIDS research.**
Use: [Sharafaldin2018], [UNB_IDS2017_Page], [UNB_IDS2018_Page], [MoustafaSlay2015], [Tavallaee2009], [Ring2019], [Goldschmidt2025_weak].
Describe the role of CICIDS2017, CSE-CIC-IDS2018, UNSW-NB15, and NSL-KDD as benchmarks. You must emphasize that they are useful for reproducibility and comparison, but they do not equate to real production traffic. Point out explicitly that CSE-CIC-IDS2018 relies on useful official documentation, although in the reviewed base an independent canonical paper equivalent to CICIDS2017's does not appear with the same clarity.

**Supervised learning and deep learning for NIDS.**
Use: [Ahmad2021], [Maseer2023], [Sarhan2021], [Corea2024].
Summarize that the dominant line in the area remains supervised classification over traffic or flows: trees, random forest, SVM, XGBoost, MLP, CNN, RNN/LSTM, hybrids. Introduce here the idea that any RL proposal must be compared to reasonable supervised baselines.

**RL/DRL fundamentals relevant to the TFG.**
Use: [SuttonBarto2018], [Mnih2015], [vanHasselt2016], [Wang2016], [Mnih2016], [Schulman2017], [Bellemare2017], [Dabney2018].
Briefly and clearly explain: state, action, reward, policy, Q-value, DQN, Double DQN, Dueling DQN, PPO, actor-critic family, and distributional RL. Present QRDQN as a distributional variant based on quantile regression.

**RL/DRL for NIDS and cyber defense.**
Use: [Alavizadeh2021], [Strickland2023], [Tellache2024], [Yang2024_weak], [Gueriani2024_weak], [Palmer2023_weak].
Distinguish two lines:
a) works that reformulate IDS detection/classification as an RL problem on labeled datasets;
b) broader autonomous cyber defense works.
You must state that the TFG is closer to line (a) than to (b). You must point out that several RL works for IDS methodologically resemble supervised reclassification with reward engineering more than rich sequential control.

**Methodological limitations of the area.**
Use: [Arp2020], [Layeghy2021], [Layeghy2022], [Cantone2024], [Ring2019].
You must be critical and mention: accuracy inflation, overly favorable random splits, leakage via spurious fields, class imbalance, lack of external validation, limited reproducibility, and poor cross-dataset generalization. This section must be one of the strongest in the chapter.

**Positioning of the TFG.**
Connect the literature with the actual implementation of the project. Use the repository information: the prototype works with a fixed canonical schema of 76 flow-based features and a 152-dimensional observation by adding a missingness mask; the environment formulates `0 = PERMIT` and `1 = BLOCK`; phase 1 is offline training/validation on public datasets; phase 2 is offline inference on private lab traffic; there is no real-time inline blocking; the main algorithm is QRDQN.
You must present the value of the TFG as follows: a concrete and reproducible formulation of a binary PERMIT/BLOCK RL defender over flows, with an internal benchmark on CICIDS2017 and an external offline validation on private lab traffic, under a prudent methodological reading.

**Permitted safe claims.**
You can state that:
- public benchmarks are necessary, though insufficient to demonstrate real deployment;
- flow-based approaches are common and pragmatic;
- there are precedents of RL for IDS;
- distributional RL like QRDQN seems underrepresented in the reviewed sample;
- the contribution of the TFG is more in the experimental formulation and prudent evaluation than in claiming absolute novelty.

**Prohibited claims.**
Do not write that:
- the TFG is "the first" in RL for NIDS;
- QRDQN is the standard of the area;
- the performance on CICIDS2017 demonstrates real efficacy;
- the laboratory phase equates to operational deployment;
- the dataset-as-environment faithfully reproduces a real network.

**Tone and Style.**
The tone must be sober, analytical, and skeptical. When a source is a preprint or the evidence is weaker, indicate this with careful language. Do not use academic marketing or grandiose statements. Prioritize terminological precision, a well-delimited scope, and a clear transition towards the experimental design of the TFG.

### Open questions and limitations of this dossier

The main documentary limitation of this review is that **several recent sources on RL for NIDS located in the exploration appear as preprints on arXiv**; therefore, in the final report they should be explicitly labeled as evidence of moderate or weak strength, not as consolidated consensus. Furthermore, **I have not verified a definitive DOI for all recent references nor a generic canonical paper for CSE-CIC-IDS2018 equivalent to CICIDS2017's**, so it is advisable for the final bibliography to accurately use official pages when necessary.

Finally, the TFG repository itself contains a **minor tension between documentation and code** regarding the exact configuration of rewards in some artifacts and invariants; this does not invalidate the project, but it does reinforce the appropriateness of tying any experimental claim to a concrete version, script, and `RUN_ID`.