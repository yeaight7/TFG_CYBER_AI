# Defensible Gap and Positioning for a Cyber Defense TFG with RL

## What is already well-studied

Intrusion detection based on ML/DL over flow representations and public benchmarks is already a very crowded field. Recent systematic reviews on anomaly-based NIDS and deep learning methods show extensive literature around binary or multiclass classification with public datasets; moreover, CICIDS2017 was created precisely as a flow-based benchmark to evaluate learning algorithms on benign traffic and several attack families. A recent survey focusing on NIDS datasets identifies 89 public datasets and notes that CIC-IDS2017 remains among the most used benchmarks by the community.

It would also not be defensible to present the use of RL in IDS/NIDS as a novelty. There are already works that reformulate detection as reinforcement learning over labeled data, even replacing the live environment with a pseudo-environment that samples recorded intrusions; others use Deep Q-Learning with classic datasets, context-aware and multi-agent architectures, transferable/adaptable frameworks, and recent proposals aimed at zero-day detection. A specific survey on DRL for NIDS reviews precisely these lines and concludes that the area is promising, but clearly existent and recognizable as a subfield.

The idea that comparability between works depends heavily on feature representation is also established. One of the consolidated criticisms of NIDS benchmarks is the absence of a standard set of features across datasets, which makes comparing methods and studying generalization difficult; hence there are proposals for common feature sets based on NetFlow or equivalents. This makes it perfectly reasonable for your TFG to use canonical flow-feature preprocessing, but it also means that this component must be sold as a methodological decision and not as a novelty in itself.

## What is partially studied but remains limited

What does appear only partially covered is the intermediate space between **"RL as a classifier on labeled datasets"** and **"RL as an autonomous mitigator in simulators/SDN"**. In several works of RL on NIDS, the action ends up corresponding to the class or label of the traffic; for example, some articles adapt the problem by making features the state and labels the actions, or replace the environment with a pseudo-environment that samples pre-recorded data. In parallel, the mitigation literature in SDN formulates actions as real countermeasures and warns that blocking merely suspicious traffic can damage network availability and functionality. From that pattern, it is reasonable to **infer** that an *offline*, *flow-level*, and explicitly *defender-centric* formulation of a binary **PERMIT/BLOCK** decision is less consolidated than either of those two poles separately.

Cost asymmetry is not new either, but it is treated unevenly. Classic IDS literature has pointed out for years that the cost of a false negative can equal the damage of a consummated attack, while a false positive imposes operational or availability costs; modern NIDS also feature cost-sensitive and imbalance-aware proposals. However, in RL the reward function continues to appear as one of the most delicate and often simplified points, and the recent survey of DRL for NIDS still highlights training efficiency and the detection of minority or unknown classes as challenges. Therefore, a *reward* explicitly biased against false negatives is defensible; what would not be defensible is selling it as an unprecedented idea or as a problem already solved by the literature.

Something similar occurs with data efficiency. The need to adapt with few samples or to new classes has already led to *few-shot/class-incremental* works and to RL proposals oriented towards adaptation and transferability with less data. But, in the reviewed literature, this usually appears as a specialized technique for adaptation or zero-day, not so much as a highly controlled experimental comparison of **learning curves RL vs. supervised** under the same feature representation, the same *splits*, and the same internal test. That comparison, posed as a methodological and experimental contribution—not algorithmic—does seem sufficiently defensible for a TFG.

## What is weakly validated in the literature

A significant part of the literature remains weakly validated regarding the quality of the benchmark itself. Engelen et al. reviewed CICIDS2017 and found problems in attack simulation, flow construction, feature extraction, and labeling; they also report that over 20% of the original traces had to be reconstructed or relabeled and that more than 25% of the flows were artifacts with no useful meaning for learning. In parallel, Lanvin et al. argue that the summarized version of CIC-IDS2017 hardly has practical import on its own. This does not invalidate your use of CICIDS2017, but it does mandate a very cautious and explicit methodological narrative about its limits.

The second weakness is evaluative leakage. In security, Arp et al. document *pitfalls* such as *data snooping*, ignoring temporal dependencies, and exploiting spurious correlations, all of which are capable of artificially inflating results; they also recommend isolating train/validation/test early on and complementing experiments on well-known datasets with more recent data from the domain. That warning fits squarely with NIDS, where Sommer and Paxson had already highlighted the high cost of errors, the lack of suitable data, and the difficulty of performing solid and operationally relevant evaluations.

The third weakness is generalization outside the same dataset. Apruzzese et al. argue that *cross-evaluation* of ML-NIDS received limited attention and propose it precisely to discover hidden risks and qualities that the intra-dataset experiment does not show. Cantone et al. are even more forceful: several classifiers achieve near-perfect results when trained and tested within the same dataset, but drop towards randomness when training and testing are separated by dataset. Furthermore, the large dataset survey notes that many groups continue to capture their own data without sharing it, something that hinders reproducibility and comparable external validation. In this context, validation with private lab traffic must be presented as an **exploratory stress test for distribution shift**, not as conclusive proof of real deployment.

Adversarial robustness and operational justification also remain fragile. Recent literature on DRL-based intrusion detection shows that these systems are also vulnerable to adversarial examples and that their robustness depends on architectural and hyperparameter choices. And, from the mitigation literature, it is emphasized that "acting" on suspicious traffic is not trivial: blocking without sufficient justification can cause operational damage. All this reinforces that your TFG should avoid any language of production, inline blocking, or operational autonomy.

## What gap can your TFG reasonably claim

The strongest and safely positionable formulation for your work is a **methodological/experimental gap**, not a "first time" or "definitive solution" gap. Your value lies not in inventing RL for IDS, but in carefully measuring what a simple defensive RL formulation contributes when the pipeline is seriously controlled, compared against supervised methods, and its risks are documented. This is a form of positioning highly aligned with what the critical literature on NIDS has been requesting for years.

**Conservative version of the gap.**
This TFG fills a gap in reproducible evaluation: applying a **QRDQN** agent to binary **PERMIT/BLOCK** decisions on pre-observed flows, with canonical feature preprocessing, a fixed *split*, and explicit leakage control, directly comparing it against at least one tabular supervised baseline on the same pipeline and measuring how results change as the training size is reduced. The contribution would not be in the "novelty of RL", but in experimental cleanliness, comparability, and the explicit statement of leakage risks and *distribution shift*.

**Balanced version of the gap.**
This TFG is situated between two existing traditions: that of RL used as a classifier on labeled datasets and that of RL used for mitigation in SDN or simulated environments. Its defensible gap is proposing an **offline, flow-level, and defender-centric** formulation of a binary **PERMIT/BLOCK** decision, with a cost-sensitive *reward* that penalizes false negatives more heavily, and contrasting it against supervised baselines under the same data pipeline, the same training *budgets*, and the same internal test. The question is not whether RL "works" in the abstract, but **when** it is worthwhile compared to Random Forest or other tabular models in this concrete framing.

**Ambitious but still defensible version of the gap.**
This TFG can claim a small protocol contribution for the evaluation of defensive RL over flow NIDS: canonical representation, dataset-as-environment setting, binary QRDQN policy, FN-sensitive *reward*, data efficiency curves, reproducible internal benchmark, and separated external validation with lab traffic. Presented this way, the work attempts to build a bridge between classification literature and *cyber defense* literature, not to claim autonomous deployment, but to honestly show what a defensive decision framing with RL gains and loses when subjected to comparison, data scarcity, and domain shift.

If I had to recommend a single formulation for the thesis, I would choose the **balanced version**: it is intellectually interesting, methodologically sound, and much less vulnerable to objections of overselling.

## What you should never claim under any circumstances

You should not claim that **"RL for IDS/NIDS has not been studied"** or that your work is the **first** application of RL to intrusion detection. That statement would directly clash with previous works on supervised pseudo-environments, DQL/DQN, context-aware systems, and adaptable/transferable frameworks.

You should not claim that your **PERMIT/BLOCK** framing equates to **real inline blocking** or an architecture ready for automatic response. The mitigation literature already warns that acting on suspicious traffic can affect availability and functionality, and classic NIDS literature insists on the high severity of errors and the gap between alert and operational action.

You should not claim that a good result on **CICIDS2017** demonstrates generalization to the real world. The literature on the dataset itself documents errors and artifacts; moreover, *cross-dataset* studies show severe performance drops outside the same benchmark.

You should not claim that your cost-sensitive treatment **solves** the FN/FP problem generally. The correct statement is that it **operationalizes** that asymmetry for your experimental setting, in line with a recognized concern in IDS and NIDS.

You should not claim that **QRDQN** is the central novelty of the work. Given current literature, it is more prudent to present it as a reasonable choice within the family of *value-based* algorithms; the originality of the TFG is better defended in the experimental design, the comparison with supervised methods, the data efficiency, and the separation between internal benchmark and external validation.

## Suggested paragraphs for the thesis

**Short version**

This work does not start from the premise that reinforcement learning is novel in intrusion detection, but rather that there is still a lack of comparative and reproducible evidence on what a simple, defensively-oriented RL formulation contributes when applied to binary **PERMIT/BLOCK** decisions over network flows. To this end, the TFG adopts a public benchmark based primarily on **CICIDS2017**, utilizes a canonical preprocessing pipeline, compares **QRDQN** against supervised baselines, and explicitly evaluates data efficiency, *leakage* risk, and sensitivity to domain shift. The results are interpreted as bounded experimental evidence, not as validation for real deployment.

**Medium version**

NIDS literature has extensively studied both supervised and deep learning approaches on public benchmarks, as well as various applications of reinforcement learning to intrusion detection. However, relevant methodological limitations persist: reliance on a single dataset, heterogeneity in feature preprocessing, scant attention to *cross-dataset* evaluation, and the risk of *leakage* or spurious correlations. In this context, the present TFG is positioned not as an absolute algorithmic novelty proposal, but as a reproducible experimental evaluation of a **QRDQN** agent for flow-level binary **PERMIT/BLOCK** decisions, compared under the same pipeline against tabular supervised models. The work pays special attention to the cost asymmetry between false negatives and false positives, data efficiency across different training sizes, and, when feasible, a separated external validation with lab-captured traffic. The latter is posed as an exploratory check for robustness against domain shift, and not as proof of production deployment.

**Formal academic version**

From a scientific positioning perspective, this TFG fits at the intersection between research on supervised learning-based NIDS and the already existing line of reinforcement learning applied to cybersecurity. The contribution of the work should not be understood as the ex nihilo introduction of RL in network intrusion, since the literature already documents formulations based on DQN/DQL, pseudo-environments built from labeled datasets, multi-agent architectures, and proposals aimed at adaptability or the detection of unseen attacks. The interest of the work lies, rather, in offering a bounded, reproducible, and methodologically prudent experimental formulation of a binary **PERMIT/BLOCK** defensive decision over network flows, supported by a public benchmark and a canonical feature extraction and normalization pipeline. On this basis, a **QRDQN** agent is compared against supervised baselines under homogeneous training and evaluation conditions, further incorporating an explicit analysis of data efficiency and a cost-sensitive consideration where false negatives receive a higher penalty than false positives. Finally, the work deliberately separates internal validation on a public benchmark from any external validation with laboratory traffic, in order to avoid confusing intra-dataset performance with operational generalization capability. Consequently, the contribution of the TFG lies primarily in the methodological and experimental realm, and not in claims of real deployment, production readiness, or real-time defensive autonomy.

## Sources supporting the gap

**Mapping claim → keys**

**Widespread use of public benchmarks and CICIDS2017:** [SHARAFALDIN2018], [YANG2022], [NID-DATA-2025], [SARHAN2022].
**RL for IDS/NIDS is already studied:** [LOPEZMARTIN2020], [SETHI2020], [ALAVIZADEH2022], [HE2024], [ALAM2025], [YANG-DRL-2026].
**FN/FP asymmetry and cost sensitivity:** [AXELSSON1999], [LEE2000], [GUPTA2022], [ATMOS2020].
**Risks of leakage and benchmark artifacts:** [ENGELEN2021], [LANVIN2023], [ARP2024], [SOMMERPAXSON2010].
**Weak generalization and the need for separated external validation:** [APRUZZESE2022], [CANTONE2024], [NID-DATA-2025].
**Data scarcity and few-shot adaptation:** [DIMONDA2024], [HE2024], [YANG-DRL-2026].

**Keys legend**

[SHARAFALDIN2018] Sharafaldin et al., *Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization*; [YANG2022] Yang et al., *A systematic literature review of methods and datasets for anomaly-based network intrusion detection*; [NID-DATA-2025] *Network Intrusion Datasets: A Survey, Limitations, and Recommendations*; [SARHAN2022] Sarhan et al., *Towards a Standard Feature Set for Network Intrusion Detection System Datasets*.

[LOPEZMARTIN2020] López-Martín et al., *Application of deep reinforcement learning to intrusion detection for supervised problems*; [SETHI2020] Sethi et al., *A context-aware robust intrusion detection system*; [ALAVIZADEH2022] Alavizadeh et al., *Deep Q-Learning Based Reinforcement Learning Approach for Network Intrusion Detection*; [HE2024] He et al., *Reinforcement Learning Meets Network Intrusion Detection*; [ALAM2025] Alam et al., *Adaptive Defense: Zero-Day Attack Detection in NIDS with Deep Reinforcement Learning*; [YANG-DRL-2026] *A Survey for Deep Reinforcement Learning Based Network Intrusion Detection*.

[ENGELEN2021] Engelen et al., *Troubleshooting an Intrusion Detection Dataset*; [LANVIN2023] Lanvin et al., *Faulty use of the CIC-IDS 2017 dataset in information security research*; [ARP2024] Arp et al., *Pitfalls in Machine Learning for Computer Security*; [SOMMERPAXSON2010] Sommer and Paxson, *Outside the Closed World*; [APRUZZESE2022] Apruzzese et al., *The Cross-evaluation of Machine Learning-based Network Intrusion Detection Systems*; [CANTONE2024] Cantone et al., *On the Cross-Dataset Generalization of Machine Learning for Network Intrusion Detection*.

[AXELSSON1999] Axelsson, *The base-rate fallacy and its implications for the difficulty of intrusion detection*; [LEE2000] Lee et al., *Toward cost-sensitive modeling for intrusion detection and response*; [GUPTA2022] Gupta et al., *CSE-IDS*; [ATMOS2020] Akbari et al., *ATMoS: Autonomous Threat Mitigation in SDN using Reinforcement Learning*; [DIMONDA2024] Di Monda et al., *Few-Shot Class-Incremental Learning for Network Intrusion Detection Systems*.

## Handoff for Codex

The following *prompt* translates the previous positioning into a thesis subsection focused on methodological rigor, comparability, and caution regarding *leakage* and *distribution shift*.

```text
Write the subsection "Justification and positioning of the work" in academic language, with an approximate length of 400 to 650 words.

TFG Context:
- Topic: "RL-based cybersecurity defender for binary PERMIT/BLOCK decisions on network flows".
- Main benchmark dataset: CICIDS2017.
- Preprocessing: canonical flow-features.
- Environment: Gymnasium-like dataset-as-environment formulation.
- Agent: QRDQN.
- Action: binary PERMIT/BLOCK space.
- Key constraint: no real-time inline blocking; PERMIT/BLOCK is an experimental abstraction of offline defensive decision-making over flows.
- Supervised baseline: at least Random Forest.
- Data efficiency experiments: 100k / 250k / 500k / 1M / 2M, all sharing the same internal test.
- External validation: private lab traffic, only if feasible, and always reported separately from the public benchmark.

Text Objective:
- Position the TFG as a methodological and experimental contribution.
- DO NOT present it as absolute novelty.
- DO NOT claim that RL for IDS doesn't exist.
- DO NOT claim real deployment, production, or operational readiness.
- DO NOT confuse internal validation on a benchmark with generalization to the real world.

Ideas that MUST appear:
1. The literature has already extensively studied ML/DL for NIDS, as well as RL for IDS/NIDS.
2. Even so, problems of comparability, leakage, benchmark artifacts, and poor cross-dataset generalization persist.
3. The TFG's gap is in reproducibly evaluating a simple, binary, defensively-oriented RL formulation over flows, cleanly comparing it with supervised methods.
4. The work emphasizes:
   - reproducible experimental design;
   - RL vs. supervised baseline comparison under the same pipeline;
   - data efficiency;
   - explicit separation between internal benchmark and external validation;
   - honest discussion of methodological risks (leakage, distribution, dataset artifacts).
5. Cost-sensitive treatment must be formulated prudently:
   - prioritizing the reduction of false negatives;
   - but not claiming the problem is universally solved.

Bibliographic keys to integrate into the text:
- [LOPEZMARTIN2020]
- [ALAVIZADEH2022]
- [HE2024]
- [SHARAFALDIN2018]
- [ENGELEN2021]
- [LANVIN2023]
- [ARP2024]
- [SOMMERPAXSON2010]
- [APRUZZESE2022]
- [CANTONE2024]
- [SARHAN2022]
- [LEE2000]
- [DIMONDA2024]

Tone:
- Sober, precise, with no marketing.
- Use expressions like "positions itself", "is framed within", "seeks to provide experimental evidence", "does not pretend to demonstrate real deployment".
- Avoid words such as:
  "novel", "state-of-the-art", "production-ready", "real-time blocking", "fully autonomous defense", "first ever".

Suggested Structure:
- Paragraph 1: State of the art and why there is no absolute novelty.
- Paragraph 2: Methodological limitations of the literature.
- Paragraph 3: Concrete gap and positioning of the TFG.
- Closing: A sentence underlining that the contribution is experimental/methodological and that external validation, if present, is interpreted as an exploration of robustness against domain shift.

If you find it appropriate, use a sentence like:
"Consequently, the work's contribution lies primarily on the methodological and experimental plane, by studying under a reproducible protocol the extent to which a binary flow-decision RL formulation can offer advantages or limitations over supervised baselines in an NIDS scenario based on public datasets."
```