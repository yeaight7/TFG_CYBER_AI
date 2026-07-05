# Methodological Evaluation for a Flow-Level RL-Based NIDS Thesis

## Executive synthesis

Your current thesis direction already aligns with the strongest methodological trend in the literature: moving away from “single benchmark, random split, high accuracy” claims and toward leakage-aware evaluation, stronger validation splits, supervised baselines, and external traffic checks. That direction is well supported. Across ML security more broadly, and NIDS specifically, the literature repeatedly shows that benchmark performance can be inflated by leakage, dataset artifacts, synthetic-scenario bias, weak split design, and under-reported experimental choices. In several recent analyses, models that score extremely well when trained and tested on the same benchmark degrade sharply under cross-dataset or out-of-domain evaluation, sometimes to performance close to chance. citeturn9view3turn32search20turn20view2turn0search2turn10view2turn12view0

For a bachelor thesis, the most defensible contribution is therefore not a claim of “real-world-ready autonomous NIDS,” but a claim of **methodologically careful internal benchmarking plus an explicit validation ladder toward realism**. In that framing, CICIDS2017 can still serve as a controlled internal benchmark; shuffled-label tests, leakage-aware feature exclusion, stricter split protocols, and supervised baselines become methodological strengths rather than side details; and lab-captured traffic becomes the key bridge from benchmark evidence to practical relevance. That position is consistent with both the older “closed-world” critique of network intrusion detection evaluation and newer work on pitfalls, generalization failure, and reproducibility in ML-for-security research. citeturn32search20turn9view3turn20view2turn0search2turn9view4

The most important expansion for your State of the Art is therefore methodological, not algorithmic: explain **why** internal benchmark results are fragile, **which metrics are operationally meaningful**, **which validation protocols are stronger than row-wise random splits**, **why reproducibility is a first-class result in cybersecurity ML**, and **what CICIDS2017 can and cannot demonstrate even when used carefully**. citeturn25search0turn17search7turn9view9turn12view0turn10view2

## Evaluation pitfalls in ML, DL, and RL NIDS

The foundational leakage concept is broader than “test contamination” in the narrow sense. Kaufman, Rosset, and Perlich define leakage as the introduction of target-related information that would not legitimately be available at prediction time, and Kapoor and Narayanan show that leakage has become a recurring source of overoptimistic claims across ML-based science. In practice, this includes fitting preprocessing on the whole dataset, performing feature selection with test information, encoding future or scenario-specific information into inputs, or constructing train/test partitions after transformations that already mixed information across the boundary. citeturn18search2turn26search12turn20view6turn29search2

In NIDS, the split design is often the most consequential methodological choice. Random row-wise splitting is attractive because it is simple and yields stable class proportions, but it does **not** guarantee independence between train and test examples. In network data, rows may share the same capture file, attack campaign, host behavior, tool output, timing regime, or day-specific context. When that happens, the model can partly learn scenario identity rather than portable intrusion semantics. The literature’s broader warning is that synthetic benchmark success does not automatically transfer to operational environments; the specific NIDS evidence is that same-dataset evaluation can look excellent while cross-dataset performance collapses. citeturn32search20turn20view2turn0search2turn9view3

This is not an abstract concern for CICIDS2017. Later studies report previously undocumented errors in attack orchestration, feature generation, documentation, and labeling, and provide corrected or recreated versions of the dataset and the extraction pipeline. The companion code/documentation for the IEEE CNS 2022 work explicitly notes duplicate-flow issues in the raw processing pipeline for the related CIC/CSE data and releases improved datasets plus a fixed CICFlowMeter variant. Lanvin and colleagues likewise report flaws including packet misorder, packet duplication, and labeling problems, with measurable downstream effects on detection performance. citeturn12view0turn12view1turn11search2turn1search3

A second pitfall is **scenario-specific artifacts**. Dube’s critique is especially useful for thesis positioning because it directly targets the common practice of building classifiers on the summarized CIC-IDS 2017 flow CSVs and then reading high classification scores as evidence of practical utility. His central conclusion is deliberately strong: models developed over the summarized data are unlikely to have practical import. Even if one does not fully adopt that strongest formulation, the paper is important because it pushes the reader to ask whether the classifier is learning attack behavior or merely learning benchmark-specific regularities embedded in the summarized flows. citeturn10view2turn10view0

A third pitfall is **overclaiming from benchmark accuracy**. Sommer and Paxson’s classic argument remains current: anomaly-based network intrusion detection is hard to validate because real benign traffic is diverse, labels are difficult to obtain, attack prevalence is low, and controlled datasets do not reproduce production conditions well. Recent work strengthens rather than weakens that critique. Arp and colleagues show that pitfalls in ML-for-security remain widespread; Layeghy and colleagues show that common synthetic NIDS datasets differ statistically from real production traffic; and Cantone and colleagues show that cross-dataset generalization can be near random chance even when within-dataset results appear nearly perfect. citeturn32search20turn9view3turn20view2turn0search2

For your thesis, the most useful interpretation is this: **a strong internal CICIDS2017 result is evidence of competence on a benchmarked task, not evidence of deployment readiness**. That is a defensible and academically strong claim. It does not diminish the thesis; it makes the thesis more credible because it aligns the claim strength with the validation evidence. citeturn10view2turn20view2turn0search2turn9view4

## Metrics and operational meaning

Accuracy should not be the headline metric for NIDS. In imbalanced settings, accuracy can remain high even when a detector misses many attacks or overwhelms analysts with false alarms. Saito and Rehmsmeier show that precision-recall analysis is more informative than ROC analysis in strongly imbalanced classification, and Davis and Goadrich show the close but importantly different perspectives offered by ROC and PR spaces under class skew. In intrusion detection, this matters because attack prevalence is typically low, so “overall correctness” can be much less informative than the quality of positive alerts. citeturn9view7turn3search17turn3search15

The confusion matrix should therefore be interpreted operationally, not just mathematically. True positives and false negatives speak to missed threat activity; false positives and true negatives speak to alerting burden and analyst time. NIST’s IDS/IDPS guidance explicitly states that no system can eliminate both false positives and false negatives at once and that reducing one often increases the other. Fawcett’s ROC primer is also helpful here because it makes clear that evaluation is about trade-offs among operating points, not a single abstract score. citeturn9view9turn9view8

For a flow-level NIDS chapter, the minimum metric set should be: **precision, recall, F1, false positive rate, false negative rate, and the confusion matrix**. Precision captures how trustworthy alerts are. Recall captures how much malicious activity the detector finds. F1 is useful as a compact balance measure, but only as a secondary summary because it hides the asymmetry between false positives and false negatives. False positive rate and false negative rate remain essential because they map more directly to operational burden and security risk. citeturn9view9turn9view8turn33search19

The base-rate fallacy is especially important for intrusion detection. Axelsson showed that in realistic low-base-rate environments, false alarm rate becomes the limiting factor for practical utility, because even a detector with good sensitivity can produce a stream of alerts dominated by false alarms when true attacks are rare. A more recent controlled experiment reinforces the operational point: when participants evaluated IDS alarms under an 86% false alarm rate instead of a 50% false alarm rate, median precision was 47% lower and time on task was 40% slower. In other words, false positives are not a cosmetic defect; they directly affect human performance. citeturn2search3turn9view10

Cost-sensitive evaluation belongs in the State of the Art because the costs of false negatives and false positives are deployment-specific. Earlier IDS evaluation work formalized expected-cost style metrics, and NIST explicitly frames tuning as choosing an acceptable trade-off between missed malicious events and analyst burden. For a bachelor thesis, you do not need to implement a full decision-theoretic evaluation to discuss this rigorously. It is enough to state that the preferred operating point depends on the defended environment, review workload, and the relative cost of missed attacks versus spurious alerts. citeturn17search7turn33search19turn9view9

A precise thesis-level wording would be: **accuracy is reported for completeness, but operational judgment should prioritize precision/recall trade-offs and explicitly discuss false-positive and false-negative consequences**. That is fully aligned with both the ML evaluation literature and the IDS-specific evaluation literature. citeturn9view7turn9view9turn2search3turn25search0

## Validation protocols and a thesis-level validation ladder

The strongest established pattern in the literature is that **validation rigor should increase as the deployment claim strengthens**. Same-dataset random or stratified splits can be acceptable for early-stage controlled benchmarking, but they mainly estimate performance under matched-distribution conditions. They do not test whether the learned decision boundary survives temporal change, new benign traffic, new institutions, or new capture conditions. That distinction is central both in general ML-for-security critiques and in recent cross-dataset NIDS studies. citeturn9view3turn32search20turn20view2turn0search2

A practical validation ladder for your thesis can therefore be framed like this. **Lowest evidential strength:** row-wise random split, even if stratified. **Stronger:** temporal split within a dataset. **Stronger still:** leave-one-day-out or leave-one-scenario-out when the dataset is organized by day, attack window, or scenario. **Higher again:** train on one dataset, test on another with a harmonized feature schema. **Highest available in a bachelor thesis:** external validation on traffic captured independently in your own lab or testbed. This ladder is partly an interpretation rather than a universally standardized protocol, but it is directly motivated by the literature on distribution shift, dataset artifacts, synthetic-vs-real mismatch, and cross-dataset failure. citeturn20view2turn0search2turn32search2turn10view2turn12view0

For CICIDS2017 specifically, **leave-one-day-out** and **leave-one-scenario-out** are especially defensible protocol proposals. They do not solve every problem, but they are much harder to game accidentally than row-wise random splits because they force separation across day-level or scenario-level context. They are therefore good methodological tools for testing whether a model has learned portable attack indicators or merely contextual regularities tied to one capture period. This is a thesis-level interpretation built on the known dataset issues, not a claim that the literature has converged on a single mandatory protocol. citeturn10view2turn12view0turn28search14turn24search2

Cross-dataset validation should be discussed as the literature’s strongest warning against benchmark overconfidence. Cantone and colleagues report that classifiers can show nearly perfect performance when trained and tested on the same dataset, yet fall to results largely commensurate with random chance under cross-dataset evaluation. Layeghy and colleagues likewise show that benchmark synthetic datasets are statistically distinct from real production traffic, which directly challenges the assumption that benchmark train/test splits are representative of deployment traffic. citeturn0search2turn20view2

External validation on lab-captured traffic is therefore not just “nice to have.” It is the clearest way to separate **internal benchmark evidence** from **evidence of robustness under independent collection conditions**. Even a small but independently captured laboratory dataset strengthens the thesis materially because it changes the question from “can the agent perform on CICIDS2017?” to “does the approach survive a change in environment, capture process, and benign traffic composition?” The literature does not imply that a lab dataset is equivalent to production traffic, but it does support the claim that independent collection is substantially stronger evidence than same-benchmark re-testing. citeturn20view2turn0search2turn32search20turn10view2

For an RL-based thesis, there is one more validation layer: stochasticity and variance. Henderson and colleagues argue that deep RL results are often difficult to interpret without multiple seeds and significance-aware reporting, and Agarwal and colleagues show that few-run evaluations can produce misleading aggregate comparisons unless uncertainty is reported explicitly. Patterson and colleagues extend this into a broader methodology guide for RL experiments, emphasizing statistical evidence, hyperparameter sensitivity, and experimenter bias. This matters for your thesis because any claim that one RL method “beats” another or “beats” a supervised baseline should be qualified by repeated runs or, at minimum, honest acknowledgement of stochastic variance. citeturn20view8turn20view9turn20view7

## Reproducibility and reporting requirements

Reproducibility is unusually important in cybersecurity ML because both the scientific and practical consequences of bad evaluation are high. Olszewski and colleagues’ large-scale study of machine learning papers in tier-1 security conferences found that substantial progress is still needed in computational reproducibility for security research, despite the fact that code, data, and measurement pipelines ought to make confirmation easier than in many non-computational disciplines. Their results also suggest that artifact-evaluated papers work at a higher rate than non-reviewed artifacts, even though artifact availability overall remains inadequate. citeturn9view4

At thesis level, the core reproducibility requirements should be explicit: dataset source and version; whether CICIDS2017 is the original CSV release, a re-extracted version, or a corrected/recreated version; exact preprocessing steps; feature schema; excluded features and the reason for each exclusion; train/validation/test partition logic; all random seeds; the saved scaler or transform objects used at inference time; experiment configuration files; and run artifacts such as predictions, confusion matrices, hyperparameters, and logs. This list is an interpretation tailored to NIDS, but it follows directly from the broader reproducibility norms reflected in the NeurIPS checklist, ACM artifact badging guidance, and the security reproducibility literature. citeturn20view12turn20view13turn9view4

A good State of the Art paragraph can also justify lightweight **dataset-card** and **model-card** style documentation. Gebru and colleagues argue for datasheets that document dataset motivation, composition, collection process, recommended uses, and maintenance. Mitchell and colleagues argue for model cards that document intended use, evaluation procedures, and performance characteristics. Adapted to cybersecurity, that means documenting attack taxonomy, traffic provenance, labeling logic, preprocessing, feature exclusions, intended deployment setting, and the presumed tolerance for false positives and false negatives. citeturn22search0turn21search1turn21search3turn23search15

For the RL part of the thesis, reproducibility has to cover **stochastic training and reporting discipline** as well as code. Multiple random seeds, identical environment settings across runs, preserved training curves, checkpointed models, and fixed evaluation scripts are not cosmetic extras; they are part of the evidence. The RL methodology literature is very clear that point estimates from one or two runs are weak grounds for comparative claims. citeturn20view8turn20view9turn20view7

A strong way to phrase this in your chapter is: **reproducibility is itself a methodological contribution in cybersecurity ML, because it constrains hidden leakage, makes evaluation auditable, and prevents fragile benchmark gains from being mistaken for scientific progress**. That statement is strongly supported by the literature. citeturn9view4turn20view6turn20view13

## CICIDS2017, positioning, and integration aids

CICIDS2017 remains widely used because it provides both raw PCAPs and derived bidirectional flow features, includes multiple attack categories, and is easy to benchmark. However, later work has shown that this convenience comes with serious caveats. Independent analyses report errors in labeling, documentation, traffic capture, and feature generation; Dube questions the practical meaning of models trained on the summarized flow data; Liu and colleagues release improved datasets and corrected labeling logic; and Rosay and colleagues propose LycoSTand and a reconstructed LYCOS-IDS2017 derived from the raw captures. A careful thesis can still use CICIDS2017, but it should do so with explicit caveats and, ideally, with awareness of corrected variants and extraction corrections. citeturn13search11turn10view2turn12view0turn12view1turn28search14turn13search1

The most defensible thesis positioning is therefore: **CICIDS2017 is an internal benchmark for controlled comparative evaluation, not a realistic proxy for deployment readiness**. Internal evidence can support claims about relative behavior under a fixed benchmark protocol, ablation results, sensitivity to preprocessing, leakage-aware design, and comparison with supervised baselines. It cannot, by itself, support strong claims about generalization to new organizations, realistic benign traffic diversity, stable analyst-facing false alarm rates, or real-world autonomous cyber defense. citeturn20view2turn0search2turn10view2turn32search20

A validation ladder is the best way to state limitations without undermining the project. You are not saying “the benchmark results are meaningless.” You are saying: **benchmark evidence is necessary but low on the evidential ladder; independent-lab evidence is stronger; production-like readiness remains outside thesis scope**. That is sober, accurate, and significantly stronger than the common benchmark-only narrative. citeturn25search0turn20view2turn0search2turn9view4

**Methodological claims to add**

- The thesis adopts a **leakage-aware evaluation posture**, treating data leakage, label leakage, temporal leakage, and scenario leakage as first-order threats to validity rather than incidental implementation issues. citeturn18search2turn20view6turn9view3
- Benchmark performance on CICIDS2017 is interpreted as **internal evidence under controlled conditions**, not as proof of cross-network or production generalization. citeturn10view2turn20view2turn0search2
- Random row-wise splits are considered weak evidence for deployment performance because they preserve matched-distribution conditions and can leak scenario regularities across partitions. citeturn32search20turn20view2turn0search2
- Precision, recall, F1, false positive rate, and false negative rate are more decision-relevant than accuracy alone for IDS/NIDS evaluation under class imbalance and low attack prevalence. citeturn9view7turn9view9turn2search3
- External validation on independently captured lab traffic is treated as the preferred next validation step because it materially increases evidential strength relative to same-benchmark testing. citeturn20view2turn0search2turn32search20
- Reproducibility is framed as part of the scientific contribution: dataset versioning, documented preprocessing, persisted transforms, fixed seeds, and saved run artifacts are required to make results auditable. citeturn9view4turn20view12turn20view13
- Because the detector is RL-based, comparative claims should acknowledge stochastic variance and avoid overinterpreting single-seed gains. citeturn20view8turn20view9turn20view7

**Pitfalls checklist**

- [ ] Did any preprocessing step fit on the full dataset instead of the training partition only? citeturn18search2turn20view6
- [ ] Are any features proxies for labels, attack windows, host identity, capture day, or file origin? citeturn9view3turn10view2turn12view0
- [ ] Were train/test rows allowed to come from the same day, scenario, or near-duplicate flow population? citeturn12view0turn11search2turn10view2
- [ ] Is the exact CICIDS2017 variant documented, including whether corrected extraction or relabeling was used? citeturn12view0turn12view1turn13search1
- [ ] Are conclusions based on accuracy alone? citeturn9view7turn2search3
- [ ] Is the operating meaning of false positives and false negatives discussed? citeturn9view9turn9view10
- [ ] Are results reported with enough detail to reproduce partitions, transforms, seeds, and model checkpoints? citeturn9view4turn20view12turn20view13
- [ ] For RL, were multiple seeds or uncertainty-aware caveats provided before claiming superiority over baselines? citeturn20view8turn20view9turn20view7

**Candidate paragraphs for integration**

> Public benchmark results in network intrusion detection should be interpreted cautiously. The literature shows that high same-dataset performance may reflect leakage, dataset artifacts, or matched-distribution evaluation rather than robust attack detection capability. This concern is especially strong in ML-based NIDS, where synthetic benchmarks differ statistically from real traffic and cross-dataset performance can degrade sharply even when within-dataset scores are near perfect. Accordingly, benchmark performance is best treated as controlled internal evidence rather than direct proof of operational readiness. citeturn9view3turn20view2turn0search2turn32search20

> Evaluation metrics for IDSs must be interpreted operationally rather than only statistically. In low-base-rate settings, accuracy can be misleading because a detector may appear strong while either missing important attacks or burdening analysts with excessive false alarms. For this reason, precision, recall, false positive rate, and false negative rate are more informative than accuracy alone, and confusion-matrix results should be read together with the expected deployment trade-off between missed attacks and alert workload. citeturn9view7turn9view9turn2search3turn33search19

> A stronger NIDS evaluation protocol should progressively increase distributional independence between training and testing data. Random row-wise splits provide only weak evidence because they preserve matched benchmark conditions. Temporal, leave-one-day-out, or leave-one-scenario-out splits are more demanding within a single benchmark, while cross-dataset and externally captured traffic provide substantially stronger evidence of robustness under distribution shift. This motivates a validation ladder in which internal benchmark testing and independent lab-captured traffic play distinct methodological roles. citeturn20view2turn0search2turn32search2turn10view2

> CICIDS2017 remains useful as a widely adopted internal benchmark, but later studies have identified substantial issues in labeling, documentation, capture quality, and feature extraction. Corrected relabeling and reconstructed feature sets have been proposed, and at least one critique argues that classifiers trained on the summarized flow data are unlikely to have direct practical import. Therefore, a careful use of CICIDS2017 can support comparative benchmarking and ablation analysis, but it cannot by itself justify strong claims about real-world generalization or deployment readiness. citeturn12view0turn12view1turn13search1turn10view2

**References to add to `references.bib`**

Compared against your attached `references.bib`, the entries below do not appear to be present under matching titles or obvious keys. The metadata was checked against publisher pages, DBLP, PubMed, JMLR, or official companion pages for the cited works. citeturn26search4turn25search5turn26search12turn9view7turn9view8turn28search0turn28search4turn24search2turn28search2turn23search10turn22search2turn20view9turn22search4turn23search15turn29search2turn22search0

```bibtex
@inproceedings{Cardenas2006IDSFramework,
  author    = {Alvaro A. C{\'a}rdenas and John S. Baras and Karl Seamon},
  title     = {A Framework for the Evaluation of Intrusion Detection Systems},
  booktitle = {Proceedings of the 2006 IEEE Symposium on Security and Privacy},
  pages     = {63--77},
  year      = {2006},
  doi       = {10.1109/SP.2006.2}
}

@article{Milenkoski2015IDSEvaluationSurvey,
  author  = {Aleksandar Milenkoski and Marco Vieira and Samuel Kounev and Alberto Avritzer and Bryan D. Payne},
  title   = {Evaluating Computer Intrusion Detection Systems: A Survey of Common Practices},
  journal = {ACM Computing Surveys},
  volume  = {48},
  number  = {1},
  pages   = {12:1--12:41},
  year    = {2015},
  doi     = {10.1145/2808691}
}

@inproceedings{Kaufman2011LeakageDataMining,
  author    = {Shachar Kaufman and Saharon Rosset and Claudia Perlich},
  title     = {Leakage in Data Mining: Formulation, Detection, and Avoidance},
  booktitle = {Proceedings of the 17th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  pages     = {556--563},
  year      = {2011},
  doi       = {10.1145/2020408.2020496}
}

@article{Fawcett2006ROC,
  author  = {Tom Fawcett},
  title   = {An Introduction to ROC Analysis},
  journal = {Pattern Recognition Letters},
  volume  = {27},
  number  = {8},
  pages   = {861--874},
  year    = {2006},
  doi     = {10.1016/j.patrec.2005.10.010}
}

@article{Saito2015PRROC,
  author  = {Takaya Saito and Marc Rehmsmeier},
  title   = {The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets},
  journal = {PLOS ONE},
  volume  = {10},
  number  = {3},
  pages   = {e0118432},
  year    = {2015},
  doi     = {10.1371/journal.pone.0118432}
}
```

```bibtex
@inproceedings{Liu2022ErrorPrevalenceNIDS,
  author    = {Lisa Liu and Gints Engelen and Timothy M. Lynar and Daryl Essam and Wouter Joosen},
  title     = {Error Prevalence in NIDS Datasets: A Case Study on CIC-IDS-2017 and CSE-CIC-IDS-2018},
  booktitle = {2022 IEEE Conference on Communications and Network Security},
  pages     = {254--262},
  year      = {2022},
  doi       = {10.1109/CNS56114.2022.9947235}
}

@inproceedings{Rosay2022CICIDS2017Analysis,
  author    = {Arnaud Rosay and Elo{\"i}se Cheval and Florent Carlier and Pascal Leroux},
  title     = {Network Intrusion Detection: A Comprehensive Analysis of CIC-IDS2017},
  booktitle = {Proceedings of the 8th International Conference on Information Systems Security and Privacy},
  pages     = {25--36},
  year      = {2022},
  doi       = {10.5220/0000157000003120}
}

@article{Dube2024FaultyCICIDS2017,
  author  = {Rohit Dube},
  title   = {Faulty Use of the CIC-IDS 2017 Dataset in Information Security Research},
  journal = {Journal of Computer Virology and Hacking Techniques},
  volume  = {20},
  number  = {1},
  pages   = {203--211},
  year    = {2024},
  doi     = {10.1007/s11416-023-00509-7}
}

@article{Layeghy2024SyntheticVsRealNIDS,
  author  = {Siamak Layeghy and Marcus Gallagher and Marius Portmann},
  title   = {Benchmarking the Benchmark -- Comparing Synthetic and Real-World Network IDS Datasets},
  journal = {Journal of Information Security and Applications},
  volume  = {80},
  pages   = {103689},
  year    = {2024},
  doi     = {10.1016/j.jisa.2023.103689}
}

@inproceedings{Olszewski2023ReproSecurityML,
  author    = {Daniel Olszewski and Allison Lu and Carson Stillman and Kevin Warren and Cole Kitroser and Alejandro Pascual and Divyajyoti Ukirde and Kevin Butler and Patrick Traynor},
  title     = {{``Get in Researchers; We're Measuring Reproducibility''}: A Reproducibility Study of Machine Learning Papers in Tier 1 Security Conferences},
  booktitle = {Proceedings of the 2023 ACM SIGSAC Conference on Computer and Communications Security},
  pages     = {3433--3459},
  year      = {2023},
  doi       = {10.1145/3576915.3623130}
}
```

```bibtex
@inproceedings{Henderson2018DRLMatters,
  author    = {Peter Henderson and Riashat Islam and Philip Bachman and Joelle Pineau and Doina Precup and David Meger},
  title     = {Deep Reinforcement Learning That Matters},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  pages     = {3207--3214},
  year      = {2018}
}

@inproceedings{Agarwal2021StatisticalPrecipice,
  author    = {Rishabh Agarwal and Max Schwarzer and Pablo Samuel Castro and Aaron Courville and Marc G. Bellemare},
  title     = {Deep Reinforcement Learning at the Edge of the Statistical Precipice},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2021}
}

@article{Patterson2024EmpiricalDesignRL,
  author  = {Andrew Patterson and Samuel Neumann and Martha White and Adam White},
  title   = {Empirical Design in Reinforcement Learning},
  journal = {Journal of Machine Learning Research},
  volume  = {25},
  pages   = {318:1--318:63},
  year    = {2024}
}

@inproceedings{Mitchell2019ModelCards,
  author    = {Margaret Mitchell and Simone Wu and Andrew Zaldivar and Parker Barnes and Lucy Vasserman and Ben Hutchinson and Elena Spitzer and Inioluwa Deborah Raji and Timnit Gebru},
  title     = {Model Cards for Model Reporting},
  booktitle = {Proceedings of the Conference on Fairness, Accountability, and Transparency},
  pages     = {220--229},
  year      = {2019},
  doi       = {10.1145/3287560.3287596}
}

@article{Gebru2021Datasheets,
  author  = {Timnit Gebru and Jamie Morgenstern and Briana Vecchione and Jennifer Wortman Vaughan and Hanna M. Wallach and Hal Daum{\'e} III and Kate Crawford},
  title   = {Datasheets for Datasets},
  journal = {Communications of the ACM},
  volume  = {64},
  number  = {12},
  pages   = {86--92},
  year    = {2021},
  doi     = {10.1145/3458723}
}
```

If you want to keep the bibliography focused, the **highest-priority methodological additions** are: Cárdenas 2006, Milenkoski 2015, Kaufman 2011, Saito and Rehmsmeier 2015, Liu et al. 2022, Dube 2024, Olszewski et al. 2023, and one RL methodology source such as Henderson 2018 or Patterson 2024. The following relevant items already appear to be covered in your current bibliography and do not need duplication unless you want stronger metadata: Sommer and Paxson 2010, Arp et al. on ML/security pitfalls, Apruzzese et al. on cross-evaluation, Engelen et al. on CICIDS2017 issues, Lanvin et al. on CICIDS2017 errors, Axelsson on the base-rate fallacy, and Cantone et al. on cross-dataset generalization. 

**Open questions and limitations**

A few details should be verified locally before final thesis integration if you plan to cite them very specifically. First, corrected CICIDS2017 variants are not a single universally canonical replacement; the literature includes relabeling fixes, CICFlowMeter fixes, and re-extracted variants such as LYCOS-IDS2017, and you should name exactly the one you use. Second, some stronger claims about attack-class-specific corruption rates or exact corrected record counts vary by paper and are better quoted only after checking the original paper you decide to cite. Third, leave-one-day-out and leave-one-scenario-out are best presented as **methodologically justified protocols for this thesis**, not as community-wide mandatory standards. citeturn12view0turn12view1turn13search1turn28search14