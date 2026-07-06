> **Superseded draft** — the maintained chapter is `report/chapters/state_of_the_art.tex`; this Markdown draft is kept for provenance only.

# State of the Art

This chapter reviews the research background for a reinforcement-learning-based defender that makes binary `PERMIT`/`BLOCK` decisions over network-flow observations. The goal is not to argue that reinforcement learning is a new idea in intrusion detection. Prior work has already explored RL and DRL for IDS, including dataset-as-environment formulations and broader autonomous cyber-defense settings [Yang2024DRLNIDSSurvey; Gueriani2024DRLIoTIDSSurvey; VERIFY: audited closest RL-IDS papers from Research2]. The goal is to position this thesis as a scoped experimental study: a QRDQN-based flow-level defender evaluated with reproducible internal benchmarking, supervised baseline comparison, data-efficiency analysis, leakage-aware validation, and planned or preferred external validation using lab-captured traffic [ProjectAgentContext; ProjectResultsSnapshot].

## 1. Network Intrusion Detection Systems

Network Intrusion Detection Systems (NIDS) monitor network activity to identify behavior that may correspond to attacks, misuse, or policy violations [ScarfoneMell2007]. They are part of a broader security-monitoring workflow: a detector observes traffic or derived events, raises alerts or labels activity, and supports later response. This distinction matters for the present thesis because the implementation studies detection and decision-making over offline flow records; it does not implement inline prevention, packet dropping, firewall updates, or operational response automation [ProjectAgentContext].

Classical NIDS research includes signature-based approaches, anomaly-based methods, specification-based methods, and hybrid systems [CITATION NEEDED]. Signature-based detection is effective when known malicious patterns can be matched, but it depends on updated signatures and is less suited to novel behavior. Anomaly-based detection instead models normal or expected traffic and flags deviations, which makes it attractive for machine learning methods but also sensitive to distribution shift, noisy labels, and changing traffic patterns [Sperotto2010FlowIDS; VERIFY:DatasetSurvey2025NIDS]. This is one reason machine learning became prominent in NIDS research: it provides a way to learn decision boundaries from labeled traffic, but it also imports familiar risks from empirical ML, including overfitting, leakage, weak validation, and poor generalization outside the training distribution [Arp2020DosDontsMLSecurity; Layeghy2023CrossDomainNIDS].

The defender studied in this thesis uses the vocabulary of access decisions: `PERMIT` for benign flows and `BLOCK` for attack flows [ProjectAgentContext]. This vocabulary is useful because it connects classification errors to operational consequences: a false negative means an attack flow would be permitted, while a false positive means a benign flow would be blocked. However, in the current implementation these are offline decisions over recorded or extracted flow rows, not active changes to network policy [ProjectAgentContext; ProjectResultsSnapshot]. The NIDS framing should therefore be understood as an experimental detection and decision framework rather than a deployed intrusion-prevention system.

## 2. Flow-Based Traffic Representation

Network traffic can be represented at several levels of granularity. Packet-level inspection retains detailed protocol and payload information, but it can be expensive, privacy-sensitive, and less practical when payloads are encrypted or unavailable [Sperotto2010FlowIDS]. Flow-based representation aggregates packets that share connection-level attributes into bidirectional flow records and summarizes them through statistics such as duration, packet counts, byte counts, inter-arrival times, packet-size distributions, TCP flags, and activity/idle behavior [Lashkari2017CICFlowMeter; Sperotto2010FlowIDS].

Flow-based NIDS is attractive because it compresses network behavior into a tabular representation that can be processed at scale by statistical, machine-learning, and deep-learning methods [Sperotto2010FlowIDS]. This representation is also compatible with widely used flow exporters and with datasets such as CICIDS2017, where PCAP traffic is converted into flow CSVs through CICFlowMeter-style feature extraction [Sharafaldin2018CICIDS2017; Lashkari2017CICFlowMeter]. For this thesis, the flow representation is the natural interface between captured traffic and the learning agent: each row is mapped into a canonical feature vector and presented as an observation [ProjectAgentContext].

The practical benefit of flow features is also their main limitation. Aggregation discards payload content and some temporal detail, so a flow record cannot express every protocol-level or sequence-level signal that may matter for attack detection [Sperotto2010FlowIDS]. Flow features can also encode dataset-specific artifacts if preprocessing is careless. For example, identifiers, raw timestamps, IP addresses, or port fields can become proxies for the label rather than generalizable indicators of malicious behavior [Arp2020DosDontsMLSecurity; ProjectAgentContext]. The project therefore defines an anti-leakage policy that excludes IP addresses, absolute timestamps, Flow IDs, unique identifiers, and direct port-proxy fields from model features [ProjectAgentContext].

The current implementation standardizes inputs through a fixed canonical schema of 76 flow features. It appends a 76-value missingness mask, where `1` means present/valid and `0` means missing/imputed, producing a 152-dimensional observation [ProjectAgentContext]. This schema is an implementation choice for reproducibility and cross-source mapping, not a claim that these 76 features are universally optimal for all NIDS tasks [ProjectAgentContext].

## 3. Public Datasets for Network Intrusion Detection

Public NIDS datasets provide shared benchmarks for comparing methods, reproducing experiments, and studying known attack scenarios. They are essential for academic work because they make it possible to evaluate different models under a common protocol [VERIFY:Sharafaldin2016DatasetFramework; VERIFY:Sharafaldin2017ReliableDataset; VERIFY:DatasetSurvey2025NIDS]. At the same time, public datasets are controlled artifacts. They reflect the traffic generators, lab topology, feature extraction tools, labels, and time period used to produce them; they should not be treated as complete substitutes for deployment traffic [VERIFY:DatasetSurvey2025NIDS; Layeghy2023CrossDomainNIDS].

KDDCup99 and NSL-KDD are historically important datasets in intrusion-detection research. They helped standardize early comparisons, and NSL-KDD attempted to reduce some redundancy problems in KDDCup99 [Tavallaee2009NSLKDD; KDDCup1999]. Their continuing presence in the literature is useful for understanding the evolution of IDS evaluation, but they are not adequate as primary evidence for modern flow-based network defense because their traffic, attack mix, and feature representation are outdated [VERIFY:DatasetSurvey2025NIDS; VERIFY:Sharafaldin2017ReliableDataset]. In this repository, NSL-KDD is explicitly treated as historical benchmarking material rather than part of the final Phase 2 simulation-facing path [ProjectAgentContext; ProjectResultsSnapshot].

UNSW-NB15 was generated in a cyber-range environment and is often used as a more modern alternative to KDD-style datasets [MoustafaSlay2015UNSWNB15]. It is relevant to this thesis for two reasons. First, it provides a point of comparison for public-dataset methodology beyond CICIDS2017. Second, because it was generated in a controlled lab environment, it supports the broader idea that external or lab-derived traffic can be useful for evaluating IDS behavior under conditions different from the internal training benchmark [MoustafaSlay2015UNSWNB15; Layeghy2023CrossDomainNIDS]. That said, UNSW-NB15 is still a benchmark dataset, not a guarantee that a model will transfer cleanly to other networks.

CICIDS2017 and CSE-CIC-IDS2018 are part of the CIC family of benchmark datasets and are widely used in recent NIDS work [Sharafaldin2018CICIDS2017; VERIFY:DatasetSurvey2025NIDS; CSECICIDS2018AWS]. CICIDS2017 is especially relevant here because it provides the main internal benchmark and the basis for the project's canonical feature schema [ProjectAgentContext]. CSE-CIC-IDS2018 is useful as related context, but its final citation should use official documentation or audited papers rather than mirrors or dataset listings [CSECICIDS2018AWS].

Modern IoT-oriented datasets such as Bot-IoT and ToN-IoT can also be useful for contextualizing the field because IoT/IIoT environments introduce different device behavior, protocol mixes, and attack scenarios [Koroniotis2019BotIoT; Alsaedi2020ToNIoT]. They should be treated as related datasets and future evaluation candidates rather than direct evidence for this project's CICIDS2017-trained model unless the project actually performs mapped cross-dataset evaluation.

## 4. CICIDS2017 as the Main Internal Benchmark

CICIDS2017 is selected as the main internal benchmark because it provides labeled traffic, attack scenarios, PCAPs, and flow CSVs produced through CICFlowMeter-style feature extraction [Sharafaldin2018CICIDS2017; Lashkari2017CICFlowMeter]. Compared with older KDD-style datasets, it was designed to address several recognized benchmark limitations, including outdated traffic patterns and restricted attack diversity [VERIFY:Sharafaldin2016DatasetFramework; VERIFY:Sharafaldin2017ReliableDataset; Sharafaldin2018CICIDS2017]. This makes it a defensible starting point for a flow-based NIDS thesis.

The dataset includes benign traffic and multiple attack families captured over a controlled time window [Sharafaldin2018CICIDS2017; VERIFY:Sharafaldin2018CICIDSAnalysis]. The exact class counts, local feature list, and preprocessing state must be verified against the local curated CSVs before final reporting [VERIFY: local CICIDS2017 curated CSV counts and schema]. In this repository, two forms of the data are described: curated CSVs tracked in `datasets/CICIDS2017/*.csv`, and raw untracked CSV exports retained locally for reference [ProjectAgentContext]. The adapter in `src/load_cicids2017.py` performs cleaning, numeric coercion, infinite/NaN handling, and canonical mapping; the code-level anti-leakage drops remain the authoritative feature gate [ProjectAgentContext].

CICIDS2017 is useful precisely because it provides a reproducible internal benchmark. It enables controlled comparison across models, splits, feature mappings, and reward configurations [ProjectResultsSnapshot]. The committed results include a best historical QRDQN run on a random split, Check A direct evaluation, Check B shuffled-label validation, and Check C hard CSV/day split [ProjectResultsSnapshot]. However, the same results snapshot also makes clear that the hardest committed generalization artifact is Check C and that leave-one-CSV-out validation has code but no committed full artifact yet [ProjectResultsSnapshot]. The internal benchmark should therefore be presented as a staged evaluation ladder rather than as a single accuracy number.

The limitations of CICIDS2017 are central to the thesis argument. The dataset is lab-generated and class-imbalanced, and performance can be inflated by favorable random splits, duplicated or near-duplicated flows, leakage-prone fields, or preprocessing choices [VERIFY:DatasetSurvey2025NIDS; Layeghy2023CrossDomainNIDS; Arp2020DosDontsMLSecurity]. A model that scores highly on a random CICIDS2017 split has not thereby demonstrated deployment readiness. This thesis should instead treat CICIDS2017 as the internal benchmark that tests whether the proposed formulation works under controlled, reproducible conditions, then separate that evidence from stricter split validation and external lab-traffic evaluation [ProjectResultsSnapshot; Layeghy2023CrossDomainNIDS].

The limitations of CICIDS2017 extend beyond its controlled lab origin. Engelen et al. conducted
a systematic reconstruction of the dataset and found that a substantial fraction of the original
traces contained artefacts that reduced their value for intrusion-detection research: flow
boundaries were sometimes incorrect, certain attack sessions were partially mislabelled, and flow
records overlapping temporal windows introduced correlation between instances [VERIFY:
Engelen2021CICIDSIssues]. Lanvin et al. further argued that the widely circulated summarised
version of CICIDS2017 is used in ways that do not reflect the dataset's original intent, and that
common evaluation practices overstate what the dataset can support [VERIFY: Lanvin2023CICIDSFaulty].
Neither critique disqualifies CICIDS2017 as a benchmark. Both require any study using it to
document its preprocessing choices, label verification steps, and the specific curated version
under evaluation.

This thesis addresses these concerns through the adapter in `src/load_cicids2017.py`, which
applies explicit cleaning and numeric coercion; through an anti-leakage policy that excludes
identifiers, absolute timestamps, IP addresses, Flow IDs, and port-proxy fields; and through
local curated CSVs tracked separately from unprocessed exports [ProjectAgentContext]. These
steps reduce the risk of the most commonly documented failure modes and make the preprocessing
choices auditable, but they do not guarantee that all benchmark artefacts are eliminated
[Arp2020DosDontsMLSecurity; ProjectResultsSnapshot].

## 5. Supervised Machine Learning and Deep Learning for NIDS

Supervised machine learning is the dominant framing for much of NIDS research: a labeled traffic record or flow is mapped to a class such as benign or attack, or to a specific attack category [LiuLang2019IDSSurvey; VERIFY:DatasetSurvey2025NIDS]. Traditional models such as Random Forest, decision trees, SVMs, logistic models, and gradient-boosted trees are common for tabular flow features, while deep learning studies explore MLPs, CNNs, RNN/LSTM models, autoencoders, attention-based models, and hybrids [LiuLang2019IDSSurvey]. These models are not merely historical baselines; on structured flow features, simpler supervised methods can be strong and sometimes difficult for more complex models to beat under fair splits [VERIFY:Sharafaldin2018CICIDSAnalysis].

Random Forest is especially important for this thesis because it is a natural baseline for tabular flow data and has been used in CICIDS2017 analysis [VERIFY:Sharafaldin2018CICIDSAnalysis]. If the thesis evaluates an RL formulation but does not compare it against strong supervised baselines, it risks attributing value to reinforcement learning when a simpler classifier might achieve similar or better performance. The repository already reflects this concern: `docs/results.md` lists a committed Random Forest baseline aligned with QRDQN splits, with measured metrics for random split (F1 attack 0.9971), day split/Check C (F1 attack 0.1446), and leave-one-out/Wednesday test (F1 attack 0.0111), backed by `runs/cicids2017/baseline_random_forest_comparison/results_rf.txt` [ProjectResultsSnapshot].

A recognised difficulty in the NIDS research community is the lack of a standard feature set
across datasets. Different tools, capture environments, and research groups extract overlapping but
non-identical flow statistics, which makes direct comparison across studies difficult. Sarhan et al.
proposed mapping CICFlowMeter-generated features to a standard set aligned with a NetFlow schema,
with the goal of improving cross-dataset comparability [VERIFY: Sarhan2022StandardFeatureSet].
This thesis does not adopt that specific standard, but the motivation is the same: a fixed
canonical mapping across all data sources — CICIDS2017 and any external lab-captured traffic —
makes the evaluation reproducible, reduces the risk of implicit feature-set leakage, and simplifies
the detection of missing or anomalous inputs through a dedicated missingness indicator [ProjectAgentContext].

The project's canonical schema maps to 76 flow features drawn from the CICIDS2017 CICFlowMeter
pipeline. Each observation is extended by a 76-value binary missingness mask, where a value of 1
indicates a present and valid feature and 0 indicates a missing or imputed value, producing a
152-dimensional input vector [ProjectAgentContext]. This schema is an implementation choice, not
a claim of universality. Its primary function is reproducibility: the same pipeline processes
both the internal CICIDS2017 benchmark and any external flow source, enabling consistent comparison
across the two data origins [ProjectAgentContext].

Deep learning NIDS studies broaden the design space, but they also illustrate common evaluation risks. Surveys report strong results across public datasets, yet many works depend on unclear splits, limited preprocessing detail, imbalanced labels, or no external validation [LiuLang2019IDSSurvey; VERIFY:DatasetSurvey2025NIDS; Arp2020DosDontsMLSecurity]. This does not mean supervised ML/DL work should be dismissed. It means the comparison must be methodological rather than rhetorical: QRDQN should be evaluated under the same data, feature mapping, split, and metrics as supervised baselines before making any algorithmic claim [ProjectResultsSnapshot; CITATION NEEDED: fair model comparison protocol source].

For that reason, this chapter positions supervised ML/DL as part of the required evidentiary standard for the thesis. The later methodology should compare the QRDQN defender against Random Forest on the canonical schema, using the same internal splits and reporting per-class metrics, false positives, and false negatives [ProjectResultsSnapshot; ProjectAgentContext]. Until those baseline artifacts exist, the State of the Art can justify the need for the comparison, but it should not claim that QRDQN outperforms supervised models.

## 6. Reinforcement Learning Background

Reinforcement learning studies how an agent selects actions in an environment to maximize accumulated reward [SuttonBarto2018RL]. The key elements are an observation or state, an action space, a reward signal, a policy that maps observations to actions, and a learning process that improves the policy from experience [SuttonBarto2018RL]. In classic control problems, the agent's actions influence future states, so the learning problem is explicitly sequential.

This thesis uses RL concepts in a narrower experimental setting. Each flow row is presented as an observation; the action is binary; the reward is computed from the relationship between the chosen action and the ground-truth label [ProjectAgentContext]. In this formulation, `0 = PERMIT / BENIGN` and `1 = BLOCK / ATTACK` [ProjectAgentContext]. A correct block on an attack and a correct permit on benign traffic are rewarded, while false positives and false negatives are penalized according to a cost-sensitive reward design [ProjectAgentContext; Lee2002CostSensitiveIDS; VERIFY:CSEIDS2021CostSensitive].

The advantage of this framing is that it makes the decision cost explicit. A supervised classifier usually optimizes a loss function that is later interpreted through metrics, whereas an RL environment can encode asymmetric false-positive and false-negative consequences directly in the reward [Lee2002CostSensitiveIDS; VERIFY:CSEIDS2021CostSensitive]. The limitation is that a static labeled dataset does not automatically provide rich temporal dynamics. Without additional state evolution, adversarial interaction, or sequential response, the RL problem can resemble reward-engineered classification [GymnasiumDocs; VERIFY: classification-as-RL prior work].

This is not a defect if it is stated honestly. The State of the Art should treat RL as the modeling framework selected for this experimental prototype, not as a guarantee of operational autonomy. The later methodology chapter can then justify the environment design, reward structure, and evaluation protocol in more detail [ProjectAgentContext].

## 7. Deep Q-Learning and Distributional Reinforcement Learning

Deep Q-Learning combines Q-learning with deep neural networks to approximate action-value functions over high-dimensional observations [Mnih2015DQN]. The original DQN work is important because it popularized two stabilizing mechanisms that remain central in value-based deep RL: experience replay, which trains from stored transitions, and target networks, which stabilize temporal-difference targets [Mnih2015DQN]. Double DQN later addressed overestimation bias by decoupling action selection from action evaluation in the target calculation [VanHasselt2016DoubleDQN]. Dueling architectures further separated state-value and action-advantage estimation [Wang2016DuelingDQN].

Distributional reinforcement learning changes the object being estimated. Instead of estimating only the expected return of an action, it models a distribution over possible returns [Bellemare2017DistributionalRL]. This matters when outcomes are uncertain or when the tail of the return distribution has decision relevance. In intrusion detection, false negatives and false positives do not have symmetric meaning, so distributional methods are conceptually attractive as part of a cost-aware decision framework. That conceptual fit should be described cautiously: the existence of a return distribution does not by itself prove better NIDS performance or risk-sensitive behavior unless the policy and evaluation actually use those properties [Bellemare2017DistributionalRL; Dabney2018QRDQN].

QRDQN, or Quantile Regression DQN, approximates the return distribution through quantiles rather than a fixed categorical support [Dabney2018QRDQN]. It is therefore a distributional variant of value-based deep RL. The project uses QRDQN as its main algorithm, implemented in a Gymnasium-compatible workflow [ProjectAgentContext; SB3ContribQRDQNDocs; GymnasiumDocs]. The correct claim is that QRDQN is a defensible experimental choice for studying binary flow-level defense under asymmetric rewards. The incorrect claim would be that QRDQN is already proven superior for NIDS; that would require same-protocol comparisons against DQN, Random Forest, and other baselines [Dabney2018QRDQN; ProjectResultsSnapshot].

At this stage, the mathematical details should remain light. The algorithm/design chapter can later introduce Bellman targets, quantile regression loss, target networks, replay buffers, and QRDQN hyperparameters. The State of the Art only needs to motivate why QRDQN belongs in the family of value-based DRL methods and why its distributional view is relevant to a cost-sensitive security decision problem [Mnih2015DQN; Bellemare2017DistributionalRL; Dabney2018QRDQN].

## 8. RL and DRL for Intrusion Detection

RL and DRL have already been applied to intrusion detection and cyber defense [Yang2024DRLNIDSSurvey; Gueriani2024DRLIoTIDSSurvey; VERIFY: audited closest RL-IDS papers from Research2]. Existing work includes formulations where the model observes dataset rows and receives rewards based on classification correctness, as well as broader scenarios where a defender agent interacts with an attacker or simulated network state [Yang2024DRLNIDSSurvey; VERIFY: AE-RL / AE-SAC sources; VERIFY: autonomous cyber defense survey]. The existence of this literature is important: the thesis contribution cannot be framed as "the first RL IDS."

Several prior works provide direct precedents for the dataset-as-environment formulation studied
in this thesis. López-Martín et al. explicitly reformulated intrusion detection as a deep RL
problem by treating labelled dataset records as environment states, class predictions as actions,
and reward as a function of classification correctness [CANDIDATE: LopezMartin2020DRLIDS, DOI
10.1016/j.eswa.2019.112963]. Their experiments on NSL-KDD and AWID compared DQN, Double DQN,
Policy Gradient, and Actor-Critic agents against supervised baselines, demonstrating that RL
agents can be made competitive with supervised classifiers under this formulation. The same group
extended this work to a dataset-as-environment formulation combining a radial basis function
network with RL-style training, evaluated across five datasets including CICIDS2017 and UNSW-NB15
[CANDIDATE: LopezMartin2021RBFOfflineRL, DOI 10.1109/ACCESS.2021.3127689]. Both works evaluate
on random train-test splits without temporal or file-based validation, and neither reports results
on external lab-traffic data. Ren et al. applied DRL jointly to feature selection and classification
on CSE-CIC-IDS2018, arguing that removing approximately 80 per cent of redundant features while
jointly training a DRL-based classifier maintained detection performance [CANDIDATE: Ren2022IDRDRL,
DOI 10.1038/s41598-022-19366-3].

A consistent pattern across this literature is that reward design is either unspecified or limited
to a symmetric correct/incorrect signal, and that the distribution of return values is not modelled
[Yang2024DRLNIDSSurvey; Gueriani2024DRLIoTIDSSurvey]. Distributional RL methods such as QR-DQN
[Dabney2018QRDQN] and C51 [Bellemare2017DistributionalRL] do not appear in the reviewed RL-IDS
literature. This makes QRDQN an exploratory algorithmic choice in this context. The thesis does
not claim that QRDQN is superior to DQN or to supervised alternatives; it evaluates whether a
QRDQN formulation can produce competitive results under a cost-sensitive reward and a reproducible
protocol, which has not been reported in the reviewed literature [Yang2024DRLNIDSSurvey;
ProjectAgentContext].

Beyond the dataset-as-environment tradition, a parallel line of work studies autonomous cyber
defence, in which RL agents learn to select active countermeasures — such as isolating hosts or
modifying routing rules — in simulated networks or attack graphs [VERIFY: ACD survey]. This thesis
is not positioned within the autonomous cyber-defence literature. The PERMIT/BLOCK decision is an
offline decision label over extracted flow records, not an active network-control action in the
current implementation [ProjectAgentContext]. The autonomous cyber-defence framing represents a
natural direction for future extension but is outside the scope of the work as implemented.

## 9. Classification-as-RL Formulation

The central methodological tension in this thesis is that labeled flow detection is naturally a supervised classification problem, yet the project formulates it as an RL environment. This can be defended, but only if the limitations are explicit. The favorable argument is that a binary security decision has asymmetric consequences: false negatives permit attacks, while false positives block benign traffic. An RL reward function can encode these consequences directly and make the trade-off visible during training [Lee2002CostSensitiveIDS; VERIFY:CSEIDS2021CostSensitive; ProjectAgentContext].

The dataset-as-environment design used in this thesis follows a pattern identified explicitly in
the RL-IDS literature. López-Martín et al. described their approach as a modification of the
standard deep RL paradigm in which the environment is not a dynamical system but a function that
samples labelled records and returns reward based on classification accuracy [CANDIDATE:
LopezMartin2020DRLIDS]. The present design applies the same conceptual model through a
Gymnasium-compatible interface [GymnasiumDocs]: each step of `RLDatasetDefenderEnv` presents
one flow observation, receives a binary PERMIT or BLOCK action, and returns a reward computed from
the ground-truth label and the asymmetric cost preference [ProjectAgentContext; Lee2002CostSensitiveIDS].

The principal justification for this framing is not that it is the only way to apply RL to
intrusion detection. It is that the binary security decision has asymmetric consequences — a
false negative permits an attack flow, while a false positive blocks benign traffic — and that a
reward function can encode this asymmetry directly rather than requiring post-hoc calibration of
a symmetric loss [Lee2002CostSensitiveIDS]. The limitation is equally clear: without temporal
dynamics, adversarial interaction, or sequential state evolution, the RL problem resembles
reward-engineered classification. This is precisely why the supervised baseline comparison is
a required element of the evaluation protocol, not an optional one [ProjectResultsSnapshot].

The second favorable argument is continuity with future sequential defense. A `PERMIT`/`BLOCK` action interface resembles the kind of decision a deployed defender might eventually make, even though the current implementation is offline [ProjectAgentContext]. By using a Gymnasium-compatible environment, the project creates a structured interface that could later be extended with temporal context, online feedback, richer actions, or simulated adversarial behavior [GymnasiumDocs; ProjectAgentContext]. This makes the formulation useful as a prototype architecture.

The critical argument is equally important. If each observation is an independent labeled row and reward simply mirrors the label, RL may add complexity without adding clear modeling power compared with supervised learning [VERIFY: classification-as-RL critique source]. Experience replay, value estimation, reward tuning, and RL hyperparameters can create additional failure modes. In such a setting, supervised baselines are not optional; they are the check that prevents the thesis from mistaking a more complex learning interface for a better detector [VERIFY:Sharafaldin2018CICIDSAnalysis; ProjectResultsSnapshot].

The correct framing is therefore balanced. This thesis does not claim that classification must be solved with RL. It studies a classification-as-RL formulation because it makes false-positive and false-negative costs explicit, aligns with a future PERMIT/BLOCK defense interface, and allows an exploratory QRDQN evaluation under a reproducible protocol [ProjectAgentContext; Lee2002CostSensitiveIDS; Dabney2018QRDQN]. The thesis should then let the experiments determine how that formulation compares with supervised baselines.

## 10. Methodological Pitfalls in ML-Based NIDS

The structural difficulty of evaluating anomaly-based NIDS under realistic conditions was
articulated by Sommer and Paxson, who argued that machine learning for network intrusion detection
is harder to validate in practice than benchmark results suggest [VERIFY: SommerPaxson2010ClosedWorld,
IEEE S&P 2010]. Their argument identifies features of the problem that controlled datasets do not
reproduce: the diversity of benign traffic in production environments, the low base rate of
genuine attacks, the challenge of obtaining representative labelled training data, and the gap
between controlled evaluation conditions and operational deployment. The implication is that high
accuracy on a public benchmark is not a reliable indicator of operational performance
[Arp2020DosDontsMLSecurity; Apruzzese2022CrossEvaluationNIDS]. This observation motivates the
evaluation design in this thesis, which separates internal benchmark performance from external
lab-traffic inference and applies staged evaluation rather than relying on a single accuracy figure.

A complementary concern is the base-rate fallacy in intrusion detection: even a classifier with
a low nominal false-positive rate can produce an operationally unacceptable number of false alarms
when attacks are rare relative to benign traffic [VERIFY: Axelsson1999BaseRate, ACM TISSEC 2000].
This follows from a straightforward Bayesian argument and provides independent justification for
reporting false-positive rate, precision, and recall alongside accuracy, and for the asymmetric
reward design in which false negatives carry a higher cost than false positives
[Lee2002CostSensitiveIDS; ProjectAgentContext]. The specific reward ratio is a scenario assumption
and should be described as such in the methodology chapter, not as a measured operational cost.

The NIDS literature contains a recurring evaluation problem: very high accuracy can be reported under conditions that do not test the intended deployment challenge [Arp2020DosDontsMLSecurity; VERIFY:DatasetSurvey2025NIDS]. Random row-wise splits can place related flows, repeated patterns, or scenario-specific artifacts in both training and test data, making the test set less independent than it appears [Layeghy2023CrossDomainNIDS; Apruzzese2022CrossEvaluationNIDS]. This is especially problematic for flow datasets where many records may share capture-period, host, or attack-generator characteristics [Arp2020DosDontsMLSecurity].

Data leakage is another central risk. If identifiers, IP addresses, absolute timestamps, Flow IDs, or ports that act as label proxies remain in the feature set, a model may learn dataset artifacts instead of attack behavior [Arp2020DosDontsMLSecurity; ProjectAgentContext]. The repository addresses this risk through an explicit anti-leakage policy and loader-level cleaning [ProjectAgentContext]. It also includes a shuffled-label validation artifact, where performance did not remain artificially high under label permutation [ProjectResultsSnapshot]. That artifact is useful evidence against obvious leakage in that historical run, but it is not exhaustive proof that every possible leakage path has been eliminated [ProjectResultsSnapshot; Arp2020DosDontsMLSecurity].

Class imbalance further complicates evaluation. In intrusion detection, some attacks are rare, and benign traffic may dominate. Accuracy can therefore hide poor minority-class detection or an unacceptable false-positive rate [VERIFY:DatasetSurvey2025NIDS; LiuLang2019IDSSurvey]. For this thesis, per-class precision, recall, F1, TP, FP, FN, TN, false-positive rate, and false-negative rate are more informative than aggregate accuracy alone [ProjectResultsSnapshot]. Attack-family error analysis should be added later if the final artifacts preserve attack-family labels [CITATION NEEDED: attack-family evaluation source].

External validity is the final major risk. Cross-dataset studies show that models with high in-dataset performance may degrade when evaluated on other datasets or traffic distributions [Layeghy2023CrossDomainNIDS; Apruzzese2022CrossEvaluationNIDS]. This supports the thesis design choice to separate internal benchmarking on CICIDS2017 from external lab-traffic validation. It also limits how strong the claims can be before that external stage is artifact-backed [ProjectResultsSnapshot].

The project already reflects a staged response to these pitfalls. Random-split results provide an internal benchmark, Check C gives a harder CSV/day split, leave-one-CSV-out exists in code but still needs a full committed artifact, and Phase 2 provides offline inference on lab-derived flow CSVs with run-specific behavior [ProjectResultsSnapshot; ProjectAgentContext]. The State of the Art should present this evaluation ladder as a methodological strength, while acknowledging the remaining gaps.

## 11. External Validation and Lab-Captured Traffic

External validation matters because public-dataset performance is not enough to establish robustness under a different traffic distribution [Layeghy2023CrossDomainNIDS; Apruzzese2022CrossEvaluationNIDS]. A model trained and tuned on CICIDS2017 may learn patterns specific to its capture environment, attack scripts, feature extraction pipeline, or class distribution [VERIFY:DatasetSurvey2025NIDS; Arp2020DosDontsMLSecurity]. Testing on traffic captured from a private lab can provide an additional distribution-shift check, especially if the same flow-extraction and canonical-mapping pipeline is used [ProjectAgentContext].

The planned or preferred Phase 2 validation in this project uses private lab traffic as offline input. The maintained pipeline extracts flow-level features, maps them to the canonical schema, loads the trained model and scaler, applies robust preprocessing options, and produces predictions plus diagnostics [ProjectAgentContext; ProjectResultsSnapshot]. This should be described as offline inference, not as live network enforcement. The current repository does not implement packet dropping, firewall rule updates, or inline blocking [ProjectAgentContext; docs/AGENT_CONTEXT.md].

Existing Phase 2 artifacts must be interpreted cautiously. The results snapshot records two benign-only Phase 2 runs with different behavior: an early run blocked all benign flows, while a later run allowed all benign flows under different conditions [ProjectResultsSnapshot]. This variation is not a failure of documentation; it is exactly why Phase 2 claims must cite exact run IDs and configurations. A low-volume or benign-only lab capture can be useful as a false-positive and domain-shift sanity check, but it is not equivalent to full external validation across benign and attack traffic [ProjectResultsSnapshot; Layeghy2023CrossDomainNIDS].

The final thesis should therefore use conditional language. If a final lab validation artifact exists, it can be described as controlled offline external-distribution validation. If only benign lab traffic exists, it should be described as a benign-traffic sanity check. If no reliable lab artifact exists, the external validation stage should remain planned or future work, and the thesis should rely on internal strict-split evidence without overstating deployment relevance [ProjectResultsSnapshot; Layeghy2023CrossDomainNIDS].

## 12. Positioning of This Thesis

The thesis is best positioned as a carefully scoped experimental prototype rather than a claim of novelty by absence of prior work. RL and DRL for intrusion detection already exist, public NIDS benchmarks are common, and supervised ML/DL methods remain strong baselines [Yang2024DRLNIDSSurvey; Gueriani2024DRLIoTIDSSurvey; VERIFY:Sharafaldin2018CICIDSAnalysis; LiuLang2019IDSSurvey]. The contribution is not "the first RL IDS" and not proof that QRDQN is the best algorithm for NIDS.

The contribution is a reproducible formulation and evaluation pipeline for a binary flow-level defender. The project maps CICIDS2017 and lab-flow inputs into a fixed 76-feature canonical schema, augments observations with a missingness mask to produce 152-dimensional inputs, exposes each row through a Gymnasium-style dataset-as-environment, and trains a QRDQN agent to choose between `PERMIT` and `BLOCK` [ProjectAgentContext; GymnasiumDocs; SB3ContribQRDQNDocs]. The design makes false-positive and false-negative consequences explicit through reward shaping, while preserving the need for supervised baseline comparison [Lee2002CostSensitiveIDS; VERIFY:CSEIDS2021CostSensitive; ProjectResultsSnapshot].

The evaluation posture should be skeptical by design. Random-split performance is useful but not sufficient; strict splits, shuffled-label validation, leave-one-CSV-out evaluation, and external lab-flow inference each test different failure modes [ProjectResultsSnapshot; Layeghy2023CrossDomainNIDS]. The thesis should report artifact-backed results where they exist and mark missing evidence clearly where it does not. In the current repository, the full leave-one-CSV-out artifact remains a gap (Random Forest baseline metrics are committed in `runs/cicids2017/baseline_random_forest_comparison/results_rf.txt`), and Phase 2 behavior must be tied to exact run artifacts [ProjectResultsSnapshot].

In concise terms, this thesis studies whether a QRDQN-based binary flow defender can be built and evaluated under a reproducible, leakage-aware methodology. Its value lies in the scoped implementation, explicit cost-sensitive PERMIT/BLOCK framing, baseline-aware experimental design, data-efficiency analysis, and separation between internal CICIDS2017 benchmarking and planned or artifact-backed external lab validation [ProjectAgentContext; ProjectResultsSnapshot; Layeghy2023CrossDomainNIDS].

## 13. Data Efficiency and Training-Scale Evaluation

An underexplored dimension of RL-based NIDS evaluation is training-data efficiency: how performance
scales with training-set size, and how much labelled data the RL formulation requires compared
with a supervised alternative under the same conditions. The RL-IDS literature surveyed in this
thesis typically trains on full public datasets without controlled ablations over data volume
[Yang2024DRLNIDSSurvey]. Research into few-shot and class-incremental NIDS has addressed data
efficiency from the supervised side, studying whether new attack families can be recognised with
minimal additional examples [VERIFY: DiMonda2024FewShotNIDS]; however, controlled training-scale
experiments — in which only the training-set size changes while all other conditions remain fixed —
are uncommon in both supervised and RL-IDS contexts.

This thesis evaluates five training budgets: 100k, 250k, 500k, 1M, and 2M samples, each using
the same internal test set and the same canonical feature schema [ProjectAgentContext]. The goal
is to characterise whether the RL agent requires substantially more data than a supervised baseline
to reach comparable performance, and to identify any regions of the learning curve where the two
approaches diverge meaningfully. These experiments are positioned as methodological evidence for
understanding the data requirements of the RL formulation, not as a benchmark against specialised
few-shot or continual-learning methods [ProjectAgentContext].
