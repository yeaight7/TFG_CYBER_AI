## 1. Executive summary

- ML/DL‑based NIDS evaluations often use **single public datasets with random train/test splits**, which can leak near‑identical flows across splits and greatly inflate accuracy and F1.[^1][^2][^3][^4]
- Legacy datasets (KDDCup99/NSL‑KDD) contain **massive redundancy and outdated attacks**, so 99%+ accuracy on them tells little about performance on modern traffic.[^5][^4][^6]
- Newer flow‑based datasets like **CICIDS2017, CSE‑CIC‑IDS2018, UNSW‑NB15, Bot‑IoT, ToN_IoT** improve realism but still have issues: class imbalance, scenario correlations, lab‑generated traffic, and inconsistent preprocessing.[^7][^8][^9][^10][^2][^3]
- Studies show **poor cross‑dataset generalization**: models tuned on one dataset (e.g. UNSW‑NB15) often degrade sharply on others (e.g. CSE‑CIC‑IDS2018, Bot‑IoT, ToN_IoT), even when within‑dataset scores are excellent.[^11][^12]
- **Feature leakage** (e.g. using IPs, ports, timestamps, or scenario‑specific fields) can let models learn dataset artefacts instead of attack behaviour, again inflating metrics.[^10][^4][^13]
- Many papers report **very high F1 (>0.99)** on CICIDS2017 using random splitting and tree‑based models, highlighting how easy it is to overfit this dataset when temporal/scenario structure is ignored.[^14][^15][^1]
- Reproducibility is weak: train/test splits, preprocessing steps, and code are often under‑specified, making it hard to reproduce or compare results fairly.[^4][^13][^16]
- Best practices emerging from recent surveys and evaluation papers include **temporal/scenario‑based splits, external validation on independent traffic (e.g. testbeds/cyber ranges), reporting per‑class metrics, and careful documentation of preprocessing and feature selection**.[^12][^17][^18][^1][^11]

For your thesis, you can credibly position a **flow‑based RL PERMIT/BLOCK agent trained on CICIDS2017 and externally validated on lab traffic** as more rigorous than the typical single‑dataset, random‑split ML/DL NIDS.

***

## 2. Dataset comparison matrix

(High‑level, focusing on NIDS evaluation aspects.)


| Dataset | Year | Traffic type | Feature type | Attack families | Size (order of magnitude) | Strengths | Weaknesses | Known evaluation risks | Relevance to my thesis | Primary citation |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| CICIDS2017 | 2017 capture, paper 2018 | Enterprise‑like network; mixed benign + modern attacks over 5 days | Bidirectional flow features (78 numeric features from CICFlowMeter + label) | Brute Force, DoS/DDoS, Heartbleed, Botnet, Web attacks, Infiltration[^2][^3] | ≈2.8M flow records[^19][^2] | Realistic traffic mix, modern attacks, full pcaps and flows; widely used; flow‑based, compatible with NetFlow‑style NIDS.[^2][^3][^1] | Highly imbalanced; some scenarios small; correlations within each day/scenario; some known labelling / format issues; design choices (timeouts, directions) implicit.[^2][^3][^14][^15] | Random per‑row splits leak flows from same attack episode into train/test; near‑duplicate flows; high reported F1 (>0.99) from simple models under random split; risk of learning scenario IDs/ports.[^1][^14][^15] | Central (main training dataset for RL agent and baseline models). | Sharafaldin et al., “Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization.”[^2][^3] |
| NSL‑KDD | 2009 | Synthetic traffic based on DARPA’98/’99; connection‑level records | Connection features (41 attributes: basic, content, traffic features) | DoS, Probe, U2R, R2L + Normal[^6] | ≈125k train, 22k test records | Fixes some KDDCup99 flaws: reduced redundancy, defined train/test; huge literature base.[^6][^5] | Outdated attacks and traffic; still synthetic; not flow‑based in the modern sense; no encrypted traffic, little realism.[^6][^5][^4] | Extreme redundancy in KDDCup99; NSL‑KDD still biased; many papers use random splits or focus only on easy DoS/Probe classes; 99%+ accuracy common.[^6][^5][^4] | Historical/contrast value; useful for background and examples of evaluation pitfalls, but not primary dataset. | Tavallaee et al., “A detailed analysis of the KDD CUP 99 data set.”[^6] |
| UNSW‑NB15 | 2015 | Hybrid real/simulated traffic from UNSW Canberra Cyber Range | Flow + packet‑inspection features (49 attributes) | 9 attack types (Fuzzers, Analysis, Backdoor, DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms)[^20][^7] | ≈2.5M records across train/test[^20][^7] | More modern than KDD/NSL; rich feature set; contains both normal and low‑footprint attacks; widely used as “challenging” benchmark.[^20][^7] | Benign traffic partly synthetic; specific to one lab topology; not as recent as CIC/IoT datasets.[^7][^17] | Predefined train/test split has different distribution; random re‑splitting can hide difficulty; risk of overfitting to lab traffic patterns.[^7][^21] | Good secondary dataset to discuss and possibly use for cross‑dataset tests or baseline comparisons. | Moustafa \& Slay, “UNSW‑NB15: a comprehensive data set for network intrusion detection systems.”[^20][^7] |
| CSE‑CIC‑IDS2018 | 2018 | Enterprise‑style network over 10 days with multiple attack scenarios | Flow features from CICFlowMeter (80+ attributes) | Brute force, DDoS, Web attacks, infiltration, botnet, etc.[^22][^23] | ≈16M flows (depending on preprocessing)[^22][^23] | Multi‑day, multi‑scenario dataset; more varied than CICIDS2017; modern traffic and attacks.[^22][^23][^10] | Complex to preprocess; heavy imbalance; documentation spread across reports; some inconsistencies in labels across days.[^17][^10] | Same risks as CICIDS2017: random splits ignore temporal/scenario structure, leading to leakage; heavy imbalance encourages accuracy inflation.[^10][^11] | Very relevant as alternative CIC‑family dataset and for discussing cross‑dataset generalization and feature consistency. | UNB/AWS “A Realistic Cyber Defense Dataset (CSE‑CIC‑IDS2018).”[^22][^23] |
| Bot‑IoT | 2018/2019 | IoT‑centric network in UNSW Cyber Range; botnet traffic vs normal | Flow/argus features (NetFlow‑like) + labels | DDoS, DoS, OS \& service scan, keylogging, data exfiltration[^8] | ≈72M flows (16.7 GB CSV full; 3M in 5% subset)[^8] | Modern IoT botnet dataset; very large; realistic lab environment; multiple attack types.[^8] | Extremely imbalanced; some attacks oversampled; lab‑specific; focus on IoT botnet, not general enterprise traffic.[^8][^10] | Random sampling and using 5% subset can hide imbalance; if sampling by row, flows from same bot attacks can be in both train and test; risk of learning IP/port patterns.[^8][^11] | Useful for future work or as example of IoT dataset; secondary relevance unless you expand beyond CICIDS2017. | Koroniotis et al., “Bot‑IoT dataset” description.[^8] |
| ToN_IoT | 2020 | Distributed IoT/IIoT/OS/network testbed (edge–fog–cloud) | Heterogeneous: network flows, telemetry, logs, OS traces[^9] | IoT‑focused intrusions and anomalies | Millions of records across multiple modalities[^9] | Heterogeneous, multi‑layer dataset; realistic IoT/IIoT testbed; good for federated/distributed IDS research.[^9] | Complex; not purely flow‑based; mapping to simple NIDS tasks is non‑trivial; lab environment. | Evaluation papers show that models trained on UNSW/Bot‑IoT struggle on ToN_IoT and vice versa, highlighting poor cross‑dataset generalization.[^9][^11] | Mainly for discussion of generalizability and future directions; not central to your current flow‑based prototype. | Moustafa et al., “Data analytics‑enabled intrusion detection: Evaluations of ToN_IoT linux datasets.”[^9] |


***

## 3. CICIDS2017 deep dive

### What CICIDS2017 is and why it’s widely used

CICIDS2017 is a **network intrusion detection evaluation dataset** created by the Canadian Institute for Cybersecurity to mimic realistic enterprise traffic and up‑to‑date attacks. It consists of full packet captures over five days (July 3–7, 2017) plus derived bidirectional flow records with labels (“BENIGN” and several attack categories). It is widely used because it addresses some weaknesses of older benchmarks (modern attacks, realistic background traffic, full pcaps, flow‑level features) and is publicly available.[^2][^3][^17][^1][^10]

### Features and what flows represent

Packet captures are processed with **CICFlowMeter** to create bidirectional flows (5‑tuple plus timeouts), each described by about 78 numeric features plus a categorical label. Features include:[^19][^3][^2]

- Basic flow statistics (duration, total packets/bytes in each direction).
- Packet size stats (min/mean/max/std).
- Time‑based features (inter‑arrival times, active/idle times).
- Flag‑based counts (e.g., FIN, SYN, URG).

Each row in the CSV represents one bidirectional flow between two endpoints over some time window, making it a natural observation for flow‑based ML and RL models.[^3][^1][^2]

### Attack categories

CICIDS2017 includes multiple attack families distributed across the five days:[^19][^2][^3]

- Brute Force (SSH/FTP).
- DoS (e.g., Hulk, Slowloris, Slowhttptest).
- DDoS (LOIC‑HTTP).
- Heartbleed.
- Botnet.
- Web attacks (XSS, SQL injection, command injection).
- Infiltration (data exfiltration from inside the network).

Benign traffic includes web browsing, email, VoIP, and other common enterprise activities.[^2][^3]

### Strengths

- **Modern, diverse attacks and realistic benign traffic** generated by human operators and realistic clients/servers.[^17][^3][^2]
- **Flow‑level representation** with a rich feature set, suitable for flow‑based NIDS and scalable ML.[^1][^3][^2]
- **Public pcaps and flows**, enabling researchers to re‑extract features or design alternative feature sets (e.g., Zeek‑based).[^1]
- Widely adopted, with many ML/DL studies and a growing ecosystem of tools and partial fixes, making it a de facto modern benchmark.[^14][^10][^17][^1]


### Limitations and known problems

- **Class imbalance**: Some attack types (e.g., infiltration) are rare, while benign and volumetric DoS flows dominate; this can hide poor minority‑class performance when only global accuracy is reported.[^10][^19][^1]
- **Scenario and temporal correlation**: Attacks are scheduled in blocks (per day/scenario), so flows within a scenario are highly correlated in time, ports, and IP addresses. Random row‑wise splitting ignores this and leaks scenario information into both train and test.[^3][^2][^1]
- **Data quality issues**: A recent case study reports **inconsistencies, corrupted records, and the need for a “fixed” version** (e.g., corrected labels, cleaned flows) to get reliable results.[^15]
- **High apparent difficulty under proper splits**: When evaluated with scenario‑/time‑aware splits and robust metrics, models perform notably worse than under random splits, highlighting over‑optimism in much of the literature.[^11][^15]


### Preprocessing concerns

Common preprocessing steps (often under‑documented) include:[^13][^4][^1]

- Drop non‑numeric fields and the “Label” column, encode labels to integers.
- Replace infinities and NaNs (e.g., “Infinity”, “NaN”) with sentinel values or zeros.
- Normalise or standardise features (min–max, z‑score).
- Drop highly correlated or near‑constant features (e.g., some Flow/Flag counters).

Risks:

- Different papers drop different subsets of features, making results incomparable.[^4][^13][^10]
- Some include fields like **source/destination IP, port, or timestamp‑derived identifiers**, which can act as shortcuts for scenario IDs rather than true attack indicators, creating **feature leakage**.[^15][^13][^4]


### Evaluation concerns

- **Random splits**: Many ML/DL‑NIDS studies report F1 ≥ 0.99 on CICIDS2017 using random splits and tree‑based methods (RF, J48, PART), showing that the dataset is trivially separable when correlation is ignored.[^14][^10][^1]
- **Scenario/temporal leakage**: If flows from the same attack campaign appear in both train and test, models can learn superficial patterns (e.g., fixed IPs/ports) instead of general attack behaviour, leading to inflated performance.[^15][^2][^3][^1]
- **Lack of external validation**: Very few works train on CICIDS2017 and test on independent datasets or real‑world traffic.[^18][^12][^4]


### How to describe CICIDS2017 cautiously in a thesis

You can describe CICIDS2017 as:

- A **widely used, modern, flow‑based IDS evaluation dataset** with realistic traffic and multiple attack families.[^17][^2][^3]
- **Suitable but imperfect**: it improves on legacy datasets but exhibits **class imbalance, scenario/temporal correlations, data quality issues, and susceptibility to random‑split leakage**.[^14][^1][^15]
- A **good first stage for offline training and internal validation**, but not sufficient alone to claim real‑world performance; hence the need for **external validation on independent lab traffic**.[^12][^11]

This framing is honest and aligns with recent evaluation and generalizability studies.

***

## 4. Leakage and evaluation risk matrix

| Risk | Description | Example in NIDS context | Why it inflates results | How to mitigate | How my thesis can address it | Sources |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| Random train/test split | Splitting individual flows randomly ignores temporal/scenario structure | Randomly shuffle all CICIDS2017 rows and use 80/20 split | Attacks present in both train and test; model sees almost identical flows in training and testing | Use temporal or scenario‑based splits; group flows by day or capture file; leave‑one‑scenario‑out | Use day‑ or scenario‑based splits for CICIDS2017 (e.g., train on some days, validate/test on others); avoid row‑wise random splits | [^2][^3][^1][^14][^15] |
| Duplicate/near‑duplicate records | Multiple identical or almost identical flows | KDDCup99 has many duplicate connections; CICIDS2017 may contain repeated or near‑identical flows from automated tools | Model memorises duplicates; cross‑validation sees same records in different folds | Deduplicate flows by key fields (5‑tuple + coarse timings) or use group splits; report if deduplication changes metrics | Consider deduplication or at least group flows by connection/attack ID when splitting; document whether deduplication is applied | [^6][^5][^15] |
| Temporal leakage | Using future flows to predict past flows in evaluation | Training on flows from July 7 (CICIDS2017) and testing on July 4 | Model learns future attack patterns; evaluation no longer reflects “predict future from past” | Ensure train timeframe < validation timeframe; design chronological splits | Design chronological splits where training uses earlier days/scenarios and testing uses later ones; clearly document timeline | [^2][^3][^1][^12] |
| Feature leakage (IPs/ports/timestamps) | Features encode scenario or dataset artefacts instead of attack behaviour | Including internal host IPs or specific destination ports that only appear in attack scenarios | Model learns “if dstIP==X then attack” rather than underlying traffic pattern; fails in new networks | Remove or mask high‑risk fields (IPs, MACs, session IDs); aggregate ports into coarse categories; test cross‑dataset | Avoid IP/MAC fields and any obvious scenario IDs; consider binning ports; evaluate on lab traffic with different addressing | [^4][^13][^15] |
| Unrealistic laboratory datasets | Lab traffic not representative of Internet‑scale networks | Simple client–server topology with scripted attacks and trivial benign workloads | Model exploits lab‑specific patterns (limited IP space, simple protocols) | Use multiple datasets and more realistic testbeds; treat lab data as one validation stage, not “the real world” | Use lab traffic as **external but still limited** validation; clearly state its controlled nature; avoid overclaiming | [^7][^8][^9][^17] |
| Poor cross‑dataset generalization | Models trained on one dataset perform poorly on others | A model trained on UNSW‑NB15 drops sharply on CSE‑CIC‑IDS2018 or Bot‑IoT | Indicates overfitting to dataset‑specific distributions and artefacts | Evaluate models across multiple datasets; design training regimes that consider domain shift | If time allows, test your agent trained on CICIDS2017 on a subset of CSE‑CIC‑IDS2018 or another flow dataset; at minimum, discuss this risk | [^11][^12][^18][^24] |
| Class imbalance | Minority attack classes under‑represented | In CICIDS2017, some attacks (e.g., infiltrations) have very few samples | High accuracy driven by majority classes; poor detection of rare but important attacks | Report per‑class metrics; use balancing techniques and cost‑sensitive losses | Use per‑class precision/recall/F1; design RL reward to penalise FN on rare attacks more heavily; maybe resample | [^19][^1][^17][^4] |
| Lack of external validation | Only internal test split from same dataset | Train and test on CICIDS2017 only | Cannot estimate performance on different networks or traffic patterns | Evaluate on independent datasets or lab‑captured traffic | Use private lab traffic as second‑stage offline validation for RL agent and baselines | [^11][^12][^25] |
| Lack of reproducibility | Incomplete reporting of splits, preprocessing, seeds | Paper says “we used 70/30 split and normalised features” with no further detail | Hard to replicate and compare; cherry‑picked splits can overstate performance | Fix and publish splits, preprocessing code, seeds; use standard folds when available | Clearly document feature set, splitting strategy, and RL environment design; ideally share code/splits | [^4][^16][^13][^26] |
| Unrealistic 99%+ metrics | Extremely high accuracy/F1, especially on legacy datasets or under random split | 99.9% accuracy on NSL‑KDD or CICIDS2017 with simple models | Often symptom of leakage, duplicates, or unbalanced metrics; misleads about deployment readiness | Use more realistic splits; emphasise per‑class metrics and cross‑dataset tests; treat extreme scores skeptically | Interpret any very high results in your own work cautiously; emphasise evaluation design over headline numbers | [^6][^5][^1][^10][^18][^24] |


***

## 5. Recommended evaluation protocol for my thesis

A rigorous but feasible protocol for your RL PERMIT/BLOCK agent and baselines:

### 5.1 Public dataset: CICIDS2017 (train/validation/internal test)

1. **Data preparation**
    - Use one of the “fixed/cleaned” CICIDS2017 CSVs if available (e.g., as used in recent robustness studies), or carefully clean NaNs/Infs and inconsistency yourself.[^1][^15]
    - Remove or mask leakage‑prone fields (IPs, MACs, timestamps that directly encode scenario), focusing on flow‑statistical features (durations, packet sizes, inter‑arrival times, flags).[^2][^3][^4]
    - Encode label as binary (BENIGN vs ATTACK) for your RL environment, but keep multi‑class labels for error analysis.
2. **Splitting strategy**
    - Design a **scenario‑ or day‑based split**, for example:
        - Train on flows from some days (e.g., Monday–Wednesday), validate on a held‑out scenario (e.g., Thursday), and test on another day (e.g., Friday).[^3][^2][^1]
    - Alternatively, **leave‑one‑scenario‑out**: for each attack scenario, train on all other scenarios and test on the held‑out one; this can be a strong robustness experiment for the State of the Art chapter.[^12][^15]
3. **RL and supervised baselines**
    - Train your **QRDQN PERMIT/BLOCK agent** on the training split, using validation to tune hyper‑parameters (learning rate, discount, network size, reward coefficients).
    - Train at least:
        - A **Random Forest** baseline on the same features and splits.[^27][^1]
        - A **supervised MLP** (deep baseline) with cross‑entropy loss.
    - Optionally, add a logistic regression and/or tree‑based baseline (e.g., J48) for reference against prior work.[^10][^1]
4. **Multiple seeds and data‑efficiency curve**
    - For each model (QRDQN, RF, MLP), run **multiple random seeds** (e.g., 3–5) to account for RL and training variability; report mean and standard deviation of metrics.[^18][^4]
    - Optionally, build a **data‑efficiency curve**: train RL and baselines on increasing fractions of the training set (e.g., 10%, 25%, 50%, 100%), to show how performance scales with data.[^1]
5. **Metrics and error analysis**
    - Report **accuracy, precision, recall, F1** for the binary PERMIT/BLOCK decision; also compute **FPR** and **FNR** since these connect to operational costs.[^4][^1]
    - Provide **confusion matrices** for internal test and external validation.[^18][^1]
    - Do **per‑class error analysis** using the original attack labels: show which attack families your PERMIT/BLOCK agent tends to miss or mislabel, even though the task is binary.

### 5.2 External validation with private lab traffic

1. **Dataset**
    - Capture flows in a controlled lab representing a small network (e.g., a few VMs and services) with both benign and attack traffic (even simple attack scripts).[^25][^28]
    - Extract flow features using a tool compatible with CICFlowMeter features (or map to a subset that overlaps reasonably with CIC features).[^2][^1]
2. **Evaluation**
    - **Do not retrain** the RL agent or supervised baselines on lab flows; use the CICIDS2017‑trained models to classify lab flows.
    - Report the same metrics (accuracy, precision, recall, F1, FPR, FNR) and confusion matrices on this external dataset.[^11][^12]
    - Qualitatively describe differences between the lab environment and CICIDS2017 (topology, services, traffic types).
3. **Interpretation**
    - Emphasise **generalization behaviour** (where and why performance drops) rather than absolute numbers.
    - Discuss whether the RL agent behaves differently (e.g., more conservative PERMIT/BLOCK) compared to supervised baselines under distribution shift.

### 5.3 Fallback plan (if lab traffic is limited)

If collecting sufficient lab‑captured traffic is not feasible:

- Use **strict public splits** such as:
    - Train on CICIDS2017, test on a subset of CSE‑CIC‑IDS2018 with compatible features, or vice versa.[^22][^23][^11]
    - Train on UNSW‑NB15, test on CICIDS2017 or a subset; interpret results cautiously due to feature mismatch.[^20][^12]
- At minimum, ensure that **no flows or scenarios overlap** between train and test, and discuss the remaining limitations honestly.

***

## 6. Claims I can safely make

You can use these claim templates in your thesis, with appropriate citations.

1. **Claim:** Public NIDS datasets such as KDDCup99 and NSL‑KDD contain large amounts of redundant records and outdated attacks, which can lead to misleadingly high evaluation scores if used in isolation.
    - **Sources:** Tavallaee2009 (NSL‑KDD), KDD99 usage review.[^6][^5][^4]
2. **Claim:** More recent datasets like UNSW‑NB15, CICIDS2017, CSE‑CIC‑IDS2018, Bot‑IoT, and ToN_IoT improve realism and attack coverage, but they still exhibit challenges such as class imbalance, scenario correlations, and dependence on specific lab environments.
    - **Sources:** UNSW‑NB15 description and evaluation, CICIDS2017 paper, dataset review.[^8][^9][^7][^20][^17][^10][^3][^2]
3. **Claim:** Many ML/DL‑based NIDS studies rely on random row‑wise train/test splits on datasets with strong temporal and scenario structure (especially CICIDS2017 and CSE‑CIC‑IDS2018), which risks leaking information between training and testing flows.
    - **Sources:** CICIDS2017 ML evaluation, CICIDS2017 robustness case study, general ML‑IDS surveys.[^13][^15][^4][^14][^1]
4. **Claim:** Empirical evidence shows that ML NIDS models trained on a single dataset often generalize poorly to other datasets such as Bot‑IoT and ToN_IoT, indicating the importance of external validation and cross‑dataset testing.
    - **Sources:** Generalizability paper (Applied Sciences 2025), ToN_IoT evaluation.[^9][^11][^12]
5. **Claim:** Reported F1 values above 0.99 on CICIDS2017 using random splits and tree‑based classifiers demonstrate that this dataset can be made trivially separable under optimistic evaluation protocols, so high metrics alone do not guarantee real‑world performance.
    - **Sources:** Evaluation of ML Techniques for Flow‑Based IDS (Sensors 2022), CICIDS2017 optimization/validation papers.[^14][^1]
6. **Claim:** Recent surveys and evaluation papers recommend temporally or scenario‑based splits, per‑class metrics, and, where possible, external validation as best practices for NIDS evaluation.
    - **Sources:** ML/DL IDS surveys, dataset review, generalizability study.[^16][^24][^13][^17][^4][^12][^18]
7. **Claim:** In this thesis, CICIDS2017 is used as a realistic yet imperfect training and internal validation dataset for a flow‑based RL defender, and a second evaluation stage on private lab traffic is introduced to partially address the lack of external validation.
    - **Sources:** CICIDS2017 paper, cyber‑range/testbed literature, generalizability and dataset reviews.[^28][^25][^17][^11][^3][^12][^2]

***

## 7. Claims I must avoid

Avoid overclaiming; examples:

- “CICIDS2017 fully represents modern real‑world network traffic.”
    - In reality, it is a controlled lab dataset with specific services, clients, and scripted attacks.[^17][^3][^2]
- “Our model works in the real world.”
    - You have offline evaluation on public datasets and lab traffic, not evidence from live production deployment.
- “99% accuracy proves the model is ready for deployment.”
    - High overall accuracy can hide poor performance on minority attack classes and may be driven by leakage or unbalanced data.[^4][^18][^1]
- “Our evaluation is leakage‑free.”
    - You can reduce leakage risk via careful splits and feature selection, but cannot guarantee absence of all hidden correlations; phrase as “we adopt split strategies designed to reduce leakage”.
- “CICIDS2017 has no known problems.”
    - There is documented evidence of label inconsistencies, corrupted flows, and the need for fixed versions.[^15]
- “The external lab dataset is fully realistic.”
    - Acknowledge that your lab traffic is still a controlled environment and only one possible external validation source.[^25][^28]

***

## 8. BibTeX candidates

```bibtex
@inproceedings{Sharafaldin2018CICIDS2017,
  author    = {Iman Sharafaldin and Arash Habibi Lashkari and Ali A. Ghorbani},
  title     = {Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization},
  booktitle = {Proceedings of the 4th International Conference on Information Systems Security and Privacy (ICISSP)},
  year      = {2018},
  url       = {https://www.unb.ca/cic/datasets/ids-2017.html}
}

@inproceedings{Moustafa2015UNSWNB15,
  author    = {Nour Moustafa and Jill Slay},
  title     = {{UNSW-NB15}: A Comprehensive Data Set for Network Intrusion Detection Systems ({UNSW-NB15} Network Data Set)},
  booktitle = {2015 Military Communications and Information Systems Conference (MilCIS)},
  year      = {2015},
  pages     = {1--6},
  doi       = {10.1109/MilCIS.2015.7348942}
}

@article{Moustafa2016UNSWEval,
  author  = {Nour Moustafa and Jill Slay},
  title   = {The Evaluation of Network Anomaly Detection Systems: Statistical Analysis of the {UNSW-NB15} Data Set and the Comparison with the {KDD99} Data Set},
  journal = {International Journal of Information and Computer Security},
  volume  = {11},
  number  = {2},
  pages   = {148--169},
  year    = {2019},
  doi     = {10.1080/19393555.2015.1125974}
}

@inproceedings{Tavallaee2009NSLKDD,
  author    = {Mahbod Tavallaee and Ebrahim Bagheri and Wei Lu and Ali A. Ghorbani},
  title     = {A Detailed Analysis of the {KDD CUP 99} Data Set},
  booktitle = {2009 IEEE Symposium on Computational Intelligence for Security and Defense Applications},
  year      = {2009},
  pages     = {1--6},
  doi       = {10.1109/CISDA.2009.5356528}
}

@article{Ozgur2016KDDUsage,
  author  = {Atilla {\"O}zg{\"u}r and Hamit Erdem},
  title   = {A Review of {KDD99} Dataset Usage in Intrusion Detection and Machine Learning Between 2010 and 2015},
  journal = {PeerJ Preprints},
  volume  = {4},
  pages   = {e1954v1},
  year    = {2016},
  doi     = {10.7287/peerj.preprints.1954v1}
}

@misc{CSECICIDS2018,
  author       = {{Canadian Institute for Cybersecurity} and {Communications Security Establishment}},
  title        = {{CSE-CIC-IDS2018}: A Realistic Cyber Defense Dataset},
  year         = {2018},
  howpublished = {\url{https://registry.opendata.aws/cse-cic-ids2018/}}
}

@misc{Koroniotis2019BotIoT,
  author       = {Nickolaos Koroniotis and Nour Moustafa and Elena Sitnikova and Benjamin Turnbull},
  title        = {Towards the Development of Realistic Botnet Dataset in the Internet of Things for Network Forensic Analytics: {Bot-IoT} Dataset},
  howpublished = {\url{https://research.unsw.edu.au/projects/bot-iot-dataset}},
  year         = {2019}
}

@inproceedings{Moustafa2020ToNIoT,
  author    = {Nour Moustafa and Mohiuddin Ahmed and others},
  title     = {Data Analytics-Enabled Intrusion Detection: Evaluations of {ToN\_IoT} Linux Datasets},
  booktitle = {2020 IEEE 19th International Conference on Trust, Security and Privacy in Computing and Communications (TrustCom)},
  year      = {2021},
  doi       = {10.1109/TrustCom50675.2020.00080}
}

@article{Thakkar2020DatasetsReview,
  author  = {Ashish Thakkar and Rachna Lohiya},
  title   = {A Review of the Advancement in Intrusion Detection Datasets},
  journal = {Procedia Computer Science},
  volume  = {167},
  pages   = {636--645},
  year    = {2020},
  doi     = {10.1016/j.procs.2020.03.270}
}

@article{Rodriguez2022FlowBasedEval,
  author  = {Mar{\'i}a Rodr{\'i}guez and {\'A}lvaro Alesanco and Lorena Mehavilla and Jos{\'e} Garc{\'i}a},
  title   = {Evaluation of Machine Learning Techniques for Traffic Flow-Based Intrusion Detection},
  journal = {Sensors},
  volume  = {22},
  number  = {23},
  pages   = {9326},
  year    = {2022},
  doi     = {10.3390/s22239326}
}

@article{Boukhamla2021CICIDS2017Validation,
  author  = {Akram Boukhamla},
  title   = {{CICIDS2017} Dataset: Performance Improvements and Validation as a Robust Intrusion Detection System Testbed},
  journal = {International Journal of Information and Computer Security},
  volume  = {15},
  number  = {1},
  pages   = {20--32},
  year    = {2021},
  doi     = {10.1504/IJICS.2021.117392}
}

@article{TrainingData2025Generalizability,
  author  = {First Author and Others}, % fill actual authors
  title   = {The Choice of Training Data and the Generalizability of Machine Learning Models for Network Intrusion Detection Systems},
  journal = {Applied Sciences},
  volume  = {15},
  number  = {15},
  pages   = {8466},
  year    = {2025},
  doi     = {10.3390/app15158466}
}
```

You can add BibTeX for more recent CICIDS2017 robustness/fix papers (e.g. the case study in ACM) once you extract the full bibliographic details.[^15]

***

## 9. Codex handoff

For drafting Spanish sections:

### Section: “Datasets de tráfico de red”

Goal: explain the main public NIDS datasets, with emphasis on flow‑based ones and their methodological implications.

Key points for Codex:

- Present KDDCup99 / NSL‑KDD briefly as **datasets históricos**, highlighting redundancy, obsolescencia de ataques y problemas de realismo.[^5][^6]
- Present UNSW‑NB15 as a **dataset de nueva generación** obtenido en un cyber range, con 9 familias de ataque y 49 características basadas en flujo, más complejo que KDD pero aún de laboratorio.[^7][^20]
- Present **CICIDS2017 y CSE‑CIC‑IDS2018** como datasets de referencia basados en flujos con tráfico y ataques modernos, describiendo: días de captura, uso de CICFlowMeter, familias de ataque y desequilibrio de clases.[^23][^22][^3][^2]
- Present **Bot‑IoT y ToN\_IoT** como datasets orientados a IoT/IIoT, generados en bancos de pruebas complejos (edge–fog–cloud).[^8][^9]
- Subrayar que estos datasets son útiles para entrenar y comparar modelos, pero **no representan totalmente el tráfico de Internet**, y deben usarse con protocolos de evaluación cuidadosos.[^11][^17]


### Section: “Limitaciones metodológicas en la evaluación de NIDS”

Goal: critically discuss the evaluation pitfalls and motivate your protocol.

Key points:

- Definir problemas como **particiones aleatorias**, duplicados, fugas temporales y de características (IP, puertos), y cómo conducen a métricas infladas (especialmente exactitud y F1 global).[^6][^5][^14][^1][^15]
- Citar estudios que muestran **generalización pobre entre datasets** (UNSW‑NB15 → CSE‑CIC‑IDS2018, Bot‑IoT, ToN\_IoT).[^9][^12][^11]
- Explicar el impacto del **desequilibrio de clases** y por qué métricas como exactitud no son suficientes; insistir en per‑class recall, F1, FPR/FNR.[^19][^4][^1]
- Mencionar la **falta de código y detalles de preprocesado** en muchos trabajos, lo que dificulta la reproducibilidad.[^16][^13][^4]
- Introducir buenas prácticas: particiones temporales o por escenario, validación cruzada por archivos de captura, uso de varios datasets y semillas, y publicación de detalles de preprocesamiento.[^12][^17][^18][^1]


### Section: “Justificación de la validación externa”

Goal: justify your second evaluation stage on private lab traffic.

Key points:

- Explicar que, según estudios de generalización y trabajos sobre cyber ranges/testbeds, **un modelo entrenado en un único dataset público no garantiza buen rendimiento en otras redes**.[^28][^9][^25][^11][^12]
- Describir brevemente qué es un **cyber range / banco de pruebas**, cómo se utiliza para generar tráfico realista y evaluar IDS.[^7][^8][^25]
- Justificar que tu tráfico de laboratorio sirve como **validación externa parcial**: mismo tipo de características (flujos) pero topología, direcciones y patrones de tráfico distintos a CICIDS2017.
- Dejar claro que la validación sigue siendo offline y en entorno controlado, por lo que no se puede afirmar que el sistema esté “listo para producción”, pero sí que se ha evaluado en **dos distribuciones de datos distintas**, algo poco frecuente en la literatura ML‑NIDS.

Style instructions:

- Usar tono académico, con transiciones suaves (“Sin embargo”, “No obstante”, “En consecuencia”).
- Evitar sobreafirmaciones; usar expresiones como “diversos estudios indican que…”, “los resultados sugieren que…”.
- Mantener coherencia terminológica: “flujo de red”, “partición temporal”, “validación externa”, “desequilibrio de clases”, “fuga de información”.
<span style="display:none">[^29][^30][^31][^32][^33][^34][^35][^36][^37][^38][^39][^40][^41]</span>

<div align="center">⁂</div>

[^1]: https://pubmed.ncbi.nlm.nih.gov/36502028/

[^2]: https://www.cs.unb.ca/research-expo/expos/2018/submissions/20180403-14-56-isharafa-at-unb.ca-toward_generating_a_new_intrusion_detection_dataset_and_intrusion_traffic_characterization.pdf

[^3]: https://www.semanticscholar.org/paper/Toward-Generating-a-New-Intrusion-Detection-Dataset-Sharafaldin-Lashkari/a27089efabc5f4abd5ddf2be2a409bff41f31199

[^4]: https://onlinelibrary.wiley.com/doi/10.1155/2023/6048087

[^5]: https://www.semanticscholar.org/paper/A-review-of-KDD99-dataset-usage-in-intrusion-and-Özgür-Erdem/d09c60cd3e493923b63ae170ad1daaf9f6cb1a69

[^6]: https://www.semanticscholar.org/paper/A-detailed-analysis-of-the-KDD-CUP-99-data-set-Tavallaee-Bagheri/fc3eb090e39d71295c362458b8a0c48d2c5d8377

[^7]: https://www.tandfonline.com/doi/abs/10.1080/19393555.2015.1125974

[^8]: https://research.unsw.edu.au/projects/bot-iot-dataset

[^9]: https://ro.ecu.edu.au/ecuworkspost2013/9883/

[^10]: https://www.ijisae.org/index.php/IJISAE/article/view/6936

[^11]: https://www.library.kab.ac.ug/Record/doaj-art-09c9f77608d34c8fa595b9714fa60ea0?sid=958771

[^12]: https://www.mdpi.com/2076-3417/15/15/8466

[^13]: http://www.jfdc.cnic.cn/EN/10.11871/jfdc.issn.2096-742X.2021.03.006

[^14]: https://www.inderscienceonline.com/doi/abs/10.1504/IJICS.2021.117392

[^15]: https://dl.acm.org/doi/fullHtml/10.1145/3600160.3605031

[^16]: https://arxiv.org/abs/2504.07839

[^17]: https://www.sciencedirect.com/science/article/pii/S1877050920307961

[^18]: https://arxiv.org/abs/2405.20038

[^19]: https://github.com/noushinpervez/Intrusion-Detection-CICIDS2017

[^20]: http://ieeexplore.ieee.org/document/7348942/

[^21]: http://www.tandfonline.com/doi/full/10.1080/19393555.2015.1125974

[^22]: https://registry.opendata.aws/cse-cic-ids2018/

[^23]: https://aws.amazon.com/marketplace/pp/prodview-qkyroawpr2aw6

[^24]: https://dl.acm.org/doi/10.1016/j.comnet.2023.110016

[^25]: https://arxiv.org/abs/2201.08473

[^26]: https://ruja.ujaen.es/items/24f750f8-1995-4c09-9f83-8d7a05aabd59

[^27]: https://www.mdpi.com/1999-4893/18/12/749

[^28]: https://ieeexplore.ieee.org/document/10491813/

[^29]: https://arxiv.org/abs/2311.06818

[^30]: https://linkinghub.elsevier.com/retrieve/pii/S1687157X26000399

[^31]: https://www.degruyterbrill.com/document/doi/10.1515/les-2026-0003/html

[^32]: https://link.springer.com/10.1007/s44217-025-01039-8

[^33]: https://dl.acm.org/doi/10.1145/3640543.3645163

[^34]: https://link.springer.com/10.1007/s10618-026-01195-x

[^35]: https://linkinghub.elsevier.com/retrieve/pii/S2352340925010339

[^36]: https://ieeexplore.ieee.org/document/8999115/

[^37]: https://dl.acm.org/doi/abs/10.1504/ijics.2021.117392

[^38]: https://pmc.ncbi.nlm.nih.gov/articles/PMC9740321/

[^39]: https://doaj.org/article/09c9f77608d34c8fa595b9714fa60ea0

[^40]: https://ouci.dntb.gov.ua/en/works/lD13LMg9/

[^41]: https://jisem-journal.com/index.php/journal/article/download/1665/653/2705

