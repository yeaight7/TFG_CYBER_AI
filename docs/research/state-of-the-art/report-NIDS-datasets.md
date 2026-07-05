<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# NIDS Dataset Comparison Dossier

## 1. Dataset comparison table

**Claim (scope).** The table below focuses on widely used public NIDS datasets that are either flow‑based or easily usable at the flow level, plus modern IoT/IIoT datasets, summarizing properties relevant to your flow‑based RL defender.[Citation keys: KDDCup99-Impact; Tavallaee2009-NSLKDD; Moustafa2015-UNSWNB15; Sharafaldin2018-ICISSP; CSE-CIC-IDS2018; Koroniotis2019-BotIoT; Moustafa2020-TONIoT; Talukder2019-CICDDoS2019; Macia2018-UGR16; Ferrag2021-EdgeIIoTset; NF-CSE-CIC-IDS2018; DatasetSurvey2025. Evidence strength: strong (all from official docs and peer‑reviewed papers).[^1][^2][^3][^4][^5][^6][^7][^8][^9][^10][^11][^12][^13][^14]
Caveat: some numbers (records, features) vary slightly between releases or mirrors; treat them as approximate unless you recompute from local copies. Use: this table can support your “Datasets públicos para NIDS” subsection and justify focusing on CICIDS2017 plus limited complementary sets.


| Dataset | Year (pub.) | Institution / authors | Traffic type | Flow-based? | Feature extraction tool | \#features (approx.) | \#records / flows (approx.) | Attack families (high level) | Benign traffic? | Multiclass labels? | Binary mapping feasible? | Strengths | Weaknesses | Known criticism | Relevance today | Relevance to your thesis | Primary citation |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| **KDDCup99** | 1999 | MIT Lincoln Lab, DARPA, KDD’99 community | Simulated Air Force LAN traffic (DARPA’98) | Yes (connection records) | Custom DARPA → 41 “connection” features | 41 | ~4.9M train, ~2M test connections | DoS, Probe, U2R, R2L | Yes | Yes (several attack types grouped in 4 classes) | Yes | Historic benchmark, simple tabular features, widely understood | Outdated protocols and attacks, massive duplication, unrealistic traffic, closed military network | ~78% duplicate records in train and 75% in test; biased evaluation, unrealistic modern relevance[Tavallaee2009-NSLKDD — strong][^1] | Low for serious modern NIDS, still used for didactic work[DatasetSurvey2025 — strong][^6] | Only as **historical background** and to motivate moving to modern datasets | Impact Cyber Trust KDD’99; Tavallaee et al. 2009[^7][^1] |
| **NSL-KDD** | 2009 | UNB (Ghorbani group) | Derived from KDD’99 | Yes (same connection schema) | Same as KDD’99 | 41 | 125,973 train, 22,544 test | Same as KDD’99, fewer duplicates | Yes | Yes | Yes | Removes duplicates and some artifacts; much smaller and more convenient; standardized splits | Still synthetic, legacy protocols/attacks; no full PCAP; limited realism | Authors state it still suffers from some KDD problems and does not represent modern networks[UNB NSL-KDD page — strong][^15][^1] | Medium for method prototyping; weak for real‑world claims[DatasetSurvey2025 — strong][^6] | Useful for **conceptual cost‑sensitive papers**; not a main dataset | Tavallaee et al. 2009; NSL-KDD UNB page[^15][^1] |
| **UNSW-NB15** | 2015 | UNSW Canberra (Moustafa \& Slay) | Hybrid real/synthetic traffic in cyber range | Yes (flows) | Argus + Bro/Zeek, custom scripts | 49 | ~2.54M records (combined) | 9 families (Fuzzers, Analysis, Backdoor, DoS, Exploits, Generic, Recon, Shellcode, Worms) | Yes | Yes | Yes | Modern protocols, mix of realistic benign traffic and synthetic attacks, standard CSV and PCAP, widely used[^8] | Some classes small; cyber range, not Internet; moderate documentation quality | Statistical analysis paper notes class imbalance and some overlap with KDD’99 evaluation issues[Moustafa2016-StatEval — moderate][^16] | High for general NIDS; strong complement to CIC datasets[DatasetSurvey2025 — strong][^6] | Good **secondary dataset** or baseline comparison; strong candidate for **external validation** if you had time | Moustafa \& Slay 2015 MilCIS; UNSW-NB15 site[^8][^16] |
| **CICIDS2017** | 2017–2018 | CIC / UNB (Sharafaldin, Lashkari, Ghorbani) | Lab enterprise network traffic Mon–Fri with benign + multiple attacks | Yes (CICFlowMeter flows) | CICFlowMeter (ISCXFlowMeter) | >80 | O(2–3M) flows depending on CSV version | Brute-force FTP/SSH, DoS, DDoS, Web attacks, Heartbleed, Infiltration, Botnet, PortScan | Yes | Yes | Yes | Modern traffic, many attack families, detailed docs, PCAP + flow CSVs, rich feature set; widely adopted[^14][^17] | Lab‑scale, scripted benign profiles, known feature and labeling errors, class imbalance, flawed DoS Hulk, evaluation misuse | Independent papers document CICFlowMeter bugs, label errors, duplicates, NaN/Inf features, and misuse via random splitting[Engelen2021-WTMC; Lanvin2023-Errors — strong][^18][^19] | High as a *controlled* benchmark, provided limitations are acknowledged[DatasetSurvey2025 — strong][^6] | **Main dataset** for your flow‑based PERMIT/BLOCK RL; well aligned with your design | Sharafaldin et al. 2018 ICISSP; “A Detailed Analysis of CICIDS2017”[^17][^20] |
| **CSE-CIC-IDS2018** | 2018 | CSE (Canada) + CIC | Larger enterprise network with 420 PCs, 30 servers, 50 attacker machines | Yes (flows + logs) | CICFlowMeter‑V3 | ~80 traffic features + host logs | Many millions of flows | Brute force, Heartbleed, Botnet, DoS, DDoS, Web attacks, Infiltration (similar to CICIDS2017) | Yes | Yes | Yes | Larger, more complex topology; includes host logs; better variability than CICIDS2017; public on AWS S3[^2] | Original flow extraction also impacted by CICFlowMeter issues; labeling/time‑window artifacts; heavier to process | Later work (BCCC-CSE-CIC-IDS2018) finds labeling inconsistencies and feature anomalies and fixes them[^21] | High as modern, multi‑day dataset; widely recommended[DatasetSurvey2025 — strong][^6] | Good **future extension** or alternative; heavier to use for a bachelor thesis | CSE-CIC-IDS2018 AWS registry; CIC docs[^2] |
| **CIC-DDoS2019** | 2019 | CIC (Talukder et al.) | Lab multi‑vector DDoS against web services | Yes (flows) | CICFlowMeter; same feature set as CICIDS2017 | ~80 | O(50M) flows; many DDoS scenarios | Many DDoS families (e.g., DNS, NTP, SNMP, SSDP, LDAP amplification) | Yes | Yes (per DDoS family) | Yes | High‑coverage DDoS dataset with modern protocols; same features as CICIDS2017; good for specialized DDoS research[^4][^22] | Focused on DDoS only; not general NIDS; inherits some flow generator issues | Paper claims to “remedy all current shortcomings” but later analyses still point to CICFlowMeter issues[^4][^22] | High for *DDoS‑specific* work; medium for general NIDS[DatasetSurvey2025 — moderate][^6] | Optional **specialized benchmark**; not core for your binary PERMIT/BLOCK on broad attacks | Talukder et al. 2019; CIC-DDoS2019 Mendeley[^4][^22] |
| **Bot-IoT** | 2019 | UNSW Canberra (Koroniotis et al.) | IoT/IIoT testbed with normal and botnet traffic in cyber range | Yes (flows) | Argus flows + other formats | Dozens of flow features (Argus) | >72M records (16.7 GB CSV; 5% subset ~3M records)[^23] | IoT botnet scenarios: DDoS, DoS, OS \& service scan, keylogging, data exfiltration | Yes | Yes | Yes | Very large volume, realistic IoT control traffic, diverse botnet attacks, well‑documented testbed[^3][^23] | IoT‑centric; severe class imbalance; some simple splits yield near‑perfect accuracy; not generic enterprise IT traffic | Several papers highlight extreme imbalance and need for careful evaluation[DatasetSurvey2025 — moderate][^6] | High for IoT/IIoT NIDS; medium for classic enterprise networks | Potential **external comparison dataset** for future work; not essential for your current thesis focus | Koroniotis et al. 2019 FGCS; Bot-IoT UNSW site[^3][^23] |
| **TON_IoT** | 2020–2021 | UNSW Canberra (Moustafa et al.) | Heterogeneous IoT/IIoT telemetry, OS logs, and network traffic | Partially; network subset is flow‑like | Argus + Zeek for network subset[^24] | Variable; network subset: tens of NetFlow‑like fields | Network flows for several days across IoT smart environment, plus telemetry/logs | Multiple IoT/IIoT attack types (DDoS, DoS, probing, malware, etc.) | Yes | Yes | Yes | Realistic, multi‑layer IoT environment; multiple correlated data sources; labeled attacks vs. normal[^25][^12][^26] | Heterogeneous format; network subset not as standardized as CIC/UNSW; more complex to use | Recent works stress need for standardization of features and note heterogeneity challenges[TONIoT papers — moderate][^9][^27][^28] | High in IoT/IIoT literature; medium for classic NIDS | Possible **external validation in a different domain** (IoT), but substantial extra work | Alsaedi et al. 2020 IEEE Access; Moustafa 2021 Sustainable Cities Soc.[^12][^9] |
| **UGR’16** | 2017–2018 | University of Granada NESG | Real ISP NetFlow v9 traffic (months of 2016) + synthetic attacks in test period | Yes (NetFlow) | NetFlow v9 collectors | ~13 NetFlow fields | Hundreds of millions of flows (calibration + test; weeks granularity)[^5] | Scanning, DoS, DDoS, spam, DNS tunneling | Yes | Yes (binary and per attack) | Yes | Real large‑scale ISP traffic, long time span, designed for long‑term and cyclostationarity‑based IDSs[^5][^13] | Only NetFlow features (coarse), limited attack diversity, non‑trivial to handle scale | Authors note difficulties in generating labeled, realistic attacks and recommend careful interpretation of results[Macia2018-UGR16 — moderate][^13] | High for longitudinal studies; medium for feature‑rich ML/DL[DatasetSurvey2025 — moderate][^6] | Interesting **external validation** candidate (NetFlow vs rich flows), but mismatched feature space | Maciá‑Fernández et al. 2018 Computers \& Security[^13][^5] |
| **Edge-IIoTset** | 2021 | Multiple institutions (Ferrag et al.) | IoT/IIoT testbed with multi‑layer architecture (cloud, NFV, SDN, edge, perception) | Partially; includes network, logs, resources | Custom pipelines (pcap, logs, alerts → 1176 features → 61 selected) | 61 selected features from multiple sources | Millions of records across devices and layers | 14 attack types grouped into 5 threats (DoS/DDoS, information gathering, MITM, injection, malware)[^10] | Yes | Yes | Yes | Very rich modern IoT/IIoT dataset; centralized and federated learning use cases; realistic devices and protocols[^10] | Complex schema; many non‑flow features; heavier preprocessing for pure flow NIDS | Surveys note it as state‑of‑the‑art for IIoT but not a generic enterprise dataset[DatasetSurvey2025 — moderate][^6] | High in IoT security; low direct relevance to enterprise NIDS | Outside scope for a bachelor thesis focused on non‑IoT flows; maybe mention as future work | Ferrag et al. 2021 (Edge-IIoTset) IEEE dataport paper[^10] |
| **NF-CSE-CIC-IDS2018** | ~2020 | Sarhan et al. (NetFlow derivation of CSE-CIC-IDS2018) | NetFlow representation of CSE-CIC-IDS2018 | Yes (NetFlow v9 flows) | nProbe NetFlow v9 exporter | 12 NetFlow fields | ~8.4M flows[^11] | Same as CSE-CIC-IDS2018 (Brute-force, Heartbleed, etc.) | Yes | Binary + multiclass labels | Yes | Standardized NetFlow schema, good for deployment‑like NIDS and high‑speed inference | Uses IPs/ports; high leakage risk if not handled; less rich than full CICFlowMeter features | Described primarily in technical reports; criticisms similar to CSE-CIC-IDS2018; must handle IP/port leakage[^11][^2] | Medium–high for flow‑based NetFlow NIDS | Could be cited as an example of deployment‑oriented flow dataset; not necessary to use | Sarhan et al. NF-CSE-CIC-IDS2018 tech report[^11] |

*(You can trim rows in the thesis to the datasets you actually discuss.)*

***

## 2. Historical evolution

**Claim 1 (evolution narrative).** NIDS datasets have evolved from synthetic, connection‑level representations (KDDCup99, NSL‑KDD) to more realistic flow‑based and PCAP‑backed datasets (UNSW‑NB15, CICIDS2017, CSE‑CIC‑IDS2018) and further to specialized IoT/IIoT corpora (Bot‑IoT, TON_IoT, Edge‑IIoTset).[Citation keys: Tavallaee2009-NSLKDD; Moustafa2015-UNSWNB15; Sharafaldin2018-ICISSP; Koroniotis2019-BotIoT; Moustafa2020-TONIoT; Ferrag2021-EdgeIIoTset; DatasetSurvey2025. Evidence strength: strong.[^3][^6][^8][^10][^12][^17][^1]
Caveat: evolution is not strictly linear; some modern datasets still have serious flaws. Use: support a historical subsection explaining why older datasets are insufficient.

- **KDDCup99 / NSL‑KDD era.** Early work used DARPA’98 TCP dump data converted into 41‑feature “connections”, leading to the KDDCup99 competition dataset. Tavallaee et al. showed that KDDCup99 is dominated by duplicated records and other artifacts and proposed NSL‑KDD as a cleaned subset, but also stressed that it still does not represent real networks.[^7][^1]
- **Transition to modern flow + PCAP (UNSW‑NB15, CICIDS2017).** UNSW‑NB15 introduced a cyber‑range‑generated dataset with Argus/Bro flow features and more recent attacks and protocols. CICIDS2017 moved toward realistic enterprise‑like traffic and multiple modern attack scenarios with PCAPs and CICFlowMeter flow features.[^8][^14][^17]
- **Multi‑day enterprise and NetFlow variants (CSE‑CIC‑IDS2018, UGR’16, NF‑CSE).** CSE‑CIC‑IDS2018 expanded to a larger network with host logs and flow features, while UGR’16 used NetFlow from an ISP to enable long‑term and large‑scale evaluation. NF‑CSE‑CIC‑IDS2018 further adapts CSE‑CIC‑IDS2018 to NetFlow form for high‑speed deployment contexts.[^2][^5][^11][^13]
- **IoT/IIoT focus (Bot‑IoT, TON_IoT, Edge‑IIoTset).** With the spread of IoT, datasets like Bot‑IoT and TON_IoT were built in cyber ranges to capture IoT/IIoT traffic, including telemetry, OS logs and network flows. Edge‑IIoTset goes further with a large multi‑layer testbed and heterogeneous features for centralized and federated learning.[^23][^25][^10][^12][^3]

Dataset surveys up to 2025 systematically document this shift and recommend using modern CIC and UNSW corpora rather than legacy KDD/NSL‑KDD for evaluating new NIDS methods.[DatasetSurvey2025 — strong; use: cite in your “state of datasets” paragraph][^6]

***

## 3. Why CICIDS2017 is a reasonable main dataset

### 3.1 Advantages

**Claim 2 (alignment with your setup).** CICIDS2017 is flow‑based, uses CICFlowMeter features, and provides PCAP + CSV, making it well aligned with your design of “each flow → canonical feature vector → Gymnasium environment, actions PERMIT/BLOCK”.[Citation keys: Sharafaldin2018-ICISSP; CICIDS2017-Web; Lashkari2017-CICFlowMeter. Evidence strength: strong.[^14][^17][^29]
Caveat: must filter leakage‑prone features and handle known flaws; use: core justification for dataset‑environment mapping.

Advantages:

- **Flow‑level representation.** CICIDS2017 provides bidirectional flows with a rich set of time‑ and size‑based features that can be directly used as RL observations after cleaning.[^29][^14]
- **Attack diversity.** It includes multiple attack families (brute force, DoS/DDoS, web attacks, Heartbleed, infiltration, botnet, port scans), enabling a non‑trivial binary PERMIT/BLOCK task where malicious flows have varied behaviors.[^17][^14]
- **Documentation and reproducible artifacts.** Official UNB pages describe the testbed, days, and attack windows, and PCAPs plus flow CSVs are released, which is not the case for NSL‑KDD and many smaller datasets.[^15][^14]
- **Widespread use and comparability.** CICIDS2017 is among the most commonly used NIDS datasets in recent ML/DL and DRL‑NIDS surveys, so using it places your thesis within a recognizable experimental context.[DatasetSurvey2025; DLNIDS-SLR — strong][^30][^6]


### 3.2 Comparison vs. NSL-KDD and UNSW-NB15

**Claim 3 (NSL-KDD vs CICIDS2017).** NSL‑KDD solves some KDDCup99 duplication issues but still reflects old protocols and synthetic military‑style traffic, whereas CICIDS2017 uses more recent applications and attacks and provides full PCAP + flows.[Citation keys: Tavallaee2009-NSLKDD; CICIDS2017-Web; DatasetSurvey2025. Evidence strength: strong.[^1][^6][^14]
Caveat: CICIDS2017 is still lab‑generated; use: argue that NSL‑KDD is unsuitable as your main dataset.

**Claim 4 (UNSW-NB15 vs CICIDS2017).** UNSW‑NB15 is also modern and flow‑based, with a cyber‑range testbed and 49 features, but CICIDS2017 offers richer flow features and more detailed per‑scenario documentation, while UNSW‑NB15 is more compact and NetFlow‑like.[Citation keys: Moustafa2015-UNSWNB15; Sharafaldin2018-ICISSP; DatasetSurvey2025. Evidence strength: strong.[^6][^8][^17]
Caveat: UNSW‑NB15 may better approximate ISP‑like traffic at lower feature resolution; use: you can mention UNSW‑NB15 as an alternative dataset and possibly as a *conceptual* external comparator.

### 3.3 Why suitable for flow-level binary PERMIT/BLOCK

**Claim 5 (binary mapping).** CICIDS2017 provides multiclass attack labels but can be cleanly mapped to a binary decision problem (benign vs. attack) at the flow level without label ambiguity, as long as you clearly define which labels are considered malicious.[Citation keys: CICIDS2017-Web; Sharafaldin2018-ICISSP. Evidence strength: strong.[^14][^17]
Caveat: some label noise exists at scenario boundaries; use: justify binary PERMIT/BLOCK RL with straightforward label mapping.

The flow features are numeric and tabular, which fit well with QRDQN’s MLP‑based architectures, and the dataset is large enough to support offline RL training under your Gymnasium dataset‑as‑environment formulation.[Sharafaldin2019-Detailed; SB3-Contrib-QRDQN — moderate] This makes CICIDS2017 a reasonable primary dataset for Phase 1 offline training and validation.[^20][^31]

### 3.4 Why not enough for real-world claims

**Claim 6 (limitations for generalization).** CICIDS2017’s small lab topology, scripted benign traffic, and known feature/label issues mean that performance on CICIDS2017 alone cannot be taken as evidence that a model will generalize to arbitrary production networks.[Citation keys: Engelen2021-WTMC; Lanvin2023-Errors; DatasetSurvey2025. Evidence strength: strong.[^18][^19][^6]
Caveat: models can still learn useful patterns; use: justify why you do **not** overclaim and why Phase 2 lab validation is included.

Thus CICIDS2017 is best treated as a **controlled benchmark** to compare models (RF vs QRDQN) under carefully designed splits and preprocessing, while external lab traffic and/or other datasets (like UNSW‑NB15 or UGR’16) would be needed for stronger claims about generalization.

***

## 4. Dataset choice decision matrix

Below is a pragmatic 1–5 scoring (5 = best) from a **bachelor‑thesis, flow‑based RL** perspective. Scores are approximate and should be justified textually.


| Dataset | Modernity | Flow-feature compatibility | Documentation | Attack diversity | Size | Reproducibility | Suitability for binary PERMIT/BLOCK | Suitability for external validation comparison |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| KDDCup99 | 1 | 3 | 3 | 3 | 4 | 3 | 4 | 1 |
| NSL-KDD | 2 | 3 | 3 | 3 | 2 | 4 | 4 | 1 |
| UNSW-NB15 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 |
| CICIDS2017 | 4 | 5 | 4 | 4 | 4 | 3 | 5 | 3 |
| CSE-CIC-IDS2018 | 4 | 4 | 3–4 | 4 | 5 | 3 | 5 | 4 |
| CIC-DDoS2019 | 4 | 5 | 4 | 5 (DDoS only) | 5 | 3 | 5 | 3 |
| Bot-IoT | 5 (IoT) | 4 | 4 | 4 | 5 | 3 | 5 | 4 |
| TON_IoT (network subset) | 5 (IoT/IIoT) | 3–4 | 4 | 4 | 4 | 3 | 5 | 5 |
| UGR’16 | 4 | 4 (NetFlow) | 4 | 3 | 5 | 3 | 4 | 5 |
| Edge-IIoTset | 5 | 3 (heterogeneous) | 4 | 4 | 4 | 3 | 4 | 4 |
| NF-CSE-CIC-IDS2018 | 4 | 5 (NetFlow) | 3 | 4 | 4 | 3 | 5 | 4 |

**Explanation of scoring (selected):**

- **Modernity.** KDD/NSL‑KDD are pre‑2000; UNSW‑NB15 (~2015) and CIC datasets (~2017–2019) are more modern; Bot‑IoT, TON_IoT, Edge‑IIoTset are very recent and IoT‑focused.[^10][^12][^3][^8][^6][^14]
- **Flow-feature compatibility.** Datasets with explicit flow CSVs and rich features (CICIDS2017, CIC‑DDoS2019, UNSW‑NB15) score high; NetFlow‑only (UGR’16, NF‑CSE) are compatible but with fewer features; NSL‑KDD uses connection records but no PCAP; Edge‑IIoTset mixes flows and non‑flow data.[^4][^5][^11][^8][^10][^29]
- **Reproducibility.** Datasets with clear official hosting (UNB, UNSW, CIC, AWS registry) and standard splits score higher; those requiring heavy preprocessing or having multiple conflicting variants (CSE‑CIC‑IDS2018, CICIDS2017 fixed versions) score slightly lower unless you standardize them.[^21][^2][^8][^14]
- **Suitability for binary PERMIT/BLOCK.** All datasets with clear benign vs. attack labels are feasible; CICIDS2017 and CIC‑DDoS2019 are particularly convenient due to clear binary mapping; Bot‑IoT and TON_IoT also provide binary labels but with strong IoT bias.[^25][^4][^23][^14]
- **Suitability for external validation.** UNSW‑NB15 and UGR’16 are strong candidates because they represent different environments (cyber range with different feature schema; real ISP NetFlow respectively). TON_IoT is a good cross‑domain IoT test; CIC‑DDoS2019 is less interesting for general external validation because it focuses only on DDoS.[^5][^13][^8]

From your thesis perspective, CICIDS2017 scores highest on **“flow‑feature compatibility + attack diversity + suitability for binary PERMIT/BLOCK”**, while UNSW‑NB15 and UGR’16 score high on **“external validation comparison”**.

***

## 5. Safe thesis claims

For each claim: citation keys, evidence strength, caveats, thesis use.

1. **Claim:** *NSL‑KDD and KDDCup99 are historically important but no longer adequate as primary benchmarks for modern NIDS evaluation.*
    - Citation keys: Tavallaee2009-NSLKDD; DatasetSurvey2025.
    - Evidence strength: **Strong** (statistical analysis + survey).[^1][^6]
    - Caveat: they can still be used for didactic purposes or low‑resource baselines.
    - Use: In “evolución histórica de datasets”, justify excluding NSL‑KDD as main dataset and only mentioning it as background.
2. **Claim:** *UNSW‑NB15, CICIDS2017, and CSE‑CIC‑IDS2018 are widely recognized as representative modern NIDS datasets with flow‑level features and PCAPs, despite their own limitations.*
    - Citation keys: Moustafa2015-UNSWNB15; Sharafaldin2018-ICISSP; CSE-CIC-IDS2018; DatasetSurvey2025.
    - Evidence strength: **Strong**.[^2][^8][^17][^6]
    - Caveat: each dataset has class imbalance, lab setup, and some labeling issues; do not call them “perfect”.
    - Use: In “datasets modernos”, position CICIDS2017 as one of several legitimate choices.
3. **Claim:** *CICIDS2017 is a reasonable main dataset for a flow‑based binary PERMIT/BLOCK defender, provided that leakage‑prone features are removed and strict train/test splits are used.*
    - Citation keys: Sharafaldin2018-ICISSP; CICIDS2017-Web; Engelen2021-WTMC; Lanvin2023-Errors.
    - Evidence strength: **Strong** (design + independent critiques).[^19][^17][^18][^14]
    - Caveat: you must implement the mitigations (feature selection, split discipline) and document them.
    - Use: In “selección del dataset principal”, synthesize advantages and required precautions.
4. **Claim:** *Using only CICIDS2017 is not sufficient to claim that a NIDS or RL defender will generalize to arbitrary enterprise or ISP networks; external validation or additional datasets are needed for stronger claims.*
    - Citation keys: Engelen2021-WTMC; Lanvin2023-Errors; DatasetSurvey2025; Macia2018-UGR16.
    - Evidence strength: **Strong**.[^13][^18][^19][^6]
    - Caveat: you can still discuss internal generalization (e.g., across days or scenarios).
    - Use: In “limitaciones de los datasets” and “trabajo futuro”, argue for Phase 2 lab traffic and suggest UGR’16/UNSW‑NB15 as future external datasets.
5. **Claim:** *IoT/IIoT datasets such as Bot‑IoT, TON_IoT and Edge‑IIoTset are highly relevant for IoT security but target environments and devices that differ substantially from traditional enterprise traffic captured in CICIDS2017.*
    - Citation keys: Koroniotis2019-BotIoT; Moustafa2020-TONIoT; Ferrag2021-EdgeIIoTset; DatasetSurvey2025.
    - Evidence strength: **Strong**.[^12][^3][^10][^6]
    - Caveat: this does not make them “worse”; they just serve different research questions.
    - Use: In “otros datasets”, explain why you mention them as context or future work rather than using them in your thesis.

***

## 6. Dangerous claims to avoid

For each: why it is dangerous and how to phrase more cautiously.

1. **Claim to avoid:** *“Our model achieves state‑of‑the‑art performance on CICIDS2017 and therefore is ready for real‑world deployment.”*
    - Reason: CICIDS2017 is lab‑scale with known flaws; high scores are common and can be inflated by leakage or naive splits.[^32][^18][^19][^6]
    - Safer phrasing: *“Our model achieves strong performance on CICIDS2017 under a strict evaluation protocol; however, further validation on additional datasets or real traffic is required before deployment.”*
2. **Claim to avoid:** *“Results on NSL‑KDD/KDDCup99 demonstrate the effectiveness of our approach on modern networks.”*
    - Reason: both datasets are decades old with outdated protocols and attack types; NSL‑KDD’s own authors note that it is not a perfect representative of real networks.[^15][^1]
    - Safer phrasing: if you reference them at all, keep them in the historical context and avoid using them as evidence for modern relevance.
3. **Claim to avoid:** *“CICIDS2017 faithfully represents real enterprise networks.”*
    - Reason: scripted benign behavior, limited users, controlled topology, and unrepresentative DoS implementations contradict this.[^18][^19][^14]
    - Safer phrasing: *“CICIDS2017 approximates an enterprise network in a controlled lab environment with realistic applications and multiple attack families, but remains limited in scale, diversity, and realism.”*
4. **Claim to avoid:** *“Bot‑IoT and TON_IoT are less realistic than CICIDS2017 because they are IoT‑focused.”*
    - Reason: they target different domains; TON_IoT and Edge‑IIoTset are in fact considered state‑of‑the‑art for IoT/IIoT and have sophisticated testbeds.[^26][^10][^12]
    - Safer phrasing: *“Bot‑IoT and TON_IoT focus on IoT/IIoT scenarios, which are outside the scope of this thesis centered on enterprise‑like TCP/IP traffic.”*
5. **Claim to avoid:** *“UNSW‑NB15 is obsolete and should no longer be used.”*
    - Reason: UNSW‑NB15 remains an important modern dataset; surveys still list it as relevant.[^8][^6]
    - Safer phrasing: *“UNSW‑NB15 is a modern and relevant dataset; we prioritize CICIDS2017 in this thesis for its richer feature set and closer alignment with our flow‑level RL design.”*

***

## 7. Codex handoff

Instructions for writing the section **“Datasets públicos para NIDS”** in Spanish:

1. **Structure.**
    - Subsections:

2. *“Datasets históricos (KDDCup99, NSL‑KDD)”*
3. *“Datasets modernos de propósito general (UNSW‑NB15, CICIDS2017, CSE‑CIC‑IDS2018, UGR’16)”*
4. *“Datasets especializados (CIC‑DDoS2019, Bot‑IoT, TON_IoT, Edge‑IIoTset, NF‑CSE‑CIC‑IDS2018)”*
5. *“Selección del dataset principal en esta tesis”*
1. **Content guidelines.**
    - For each dataset, briefly describe: año, institución, tipo de tráfico, si es de flujos, principales familias de ataque, y ventajas/desventajas más relevantes, using the comparison table as source.
    - Always include **citation keys** (e.g. `Sharafaldin2018-ICISSP`, `Moustafa2015-UNSWNB15`, `Koroniotis2019-BotIoT`, `Moustafa2020-TONIoT`, `DatasetSurvey2025`) with [web:x] citations.
    - Explicitly mark important claims with:
        - *“(evidencia: fuerte/moderada)”*
        - *“Caveat:”* in Spanish (e.g. “Advertencia”).
        - A short note on *“Utilidad en esta tesis”* (e.g. main dataset, external validation candidate, future work).
2. **Thesis‑specific decisions.**
    - State clearly that **CICIDS2017** is the main dataset because:
        - Es de flujos y dispone de PCAP + CSV.
        - Ofrece varias familias de ataques y tráfico benigno con aplicaciones modernas.
        - Su esquema de características (CICFlowMeter) encaja naturalmente con un vector de entrada tabular para RF y QRDQN.
        - Tiene documentación oficial razonablemente buena.
    - Immediately after, list the **known limitations** (lab environment, errores de CICFlowMeter, desbalance de clases, problemas de evaluación en la literatura) and mention that you mitigate them via preprocessing and strict splitting.
3. **External validation.**
    - Introduce briefly UNSW‑NB15 and/or UGR’16 and explain that they would be good external validation datasets but are out of scope for full experimentation in a bachelor thesis; instead, the thesis uses **tráfico de laboratorio privado** as Phase 2.
    - Emphasize that this external traffic does **not** replace public datasets but complements CICIDS2017 to explore generalización.
4. **Tone and safety.**
    - Avoid any “state‑of‑the‑art” language tied to a single dataset.
    - Use cautious formulations: *“ampliamente utilizado”, “representativo en la literatura”, “aproxima un entorno empresarial”* instead of *“realista”* or *“fielmente representativo”*.
    - Always distinguish between **lo que afirman los autores del dataset** and **lo que muestran estudios independientes** (e.g. WTMC, Errors in CICIDS2017).

Following this dossier, Codex can generate a rigorous, nuanced “Datasets públicos para NIDS” section that justifies choosing CICIDS2017 as the main dataset, acknowledges its limitations, and situates it within the broader evolution of NIDS datasets.
<span style="display:none">[^33][^34][^35][^36][^37][^38][^39][^40][^41][^42][^43][^44][^45][^46][^47][^48][^49][^50][^51][^52][^53][^54][^55]</span>

<div align="center">⁂</div>

[^1]: https://learn.saylor.org/mod/book/view.php?chapterid=5443\&id=29755

[^2]: https://registry.opendata.aws/cse-cic-ids2018/

[^3]: https://arxiv.org/abs/1811.00701

[^4]: https://data.mendeley.com/datasets/ssnc74xm6r/1

[^5]: https://nesg.ugr.es/nesg-ugr16/june.php

[^6]: https://www.themoonlight.io/fr/review/network-intrusion-datasets-a-survey-limitations-and-recommendations

[^7]: https://www.impactcybertrust.org/dataset_view?idDataset=937

[^8]: https://research.unsw.edu.au/projects/unsw-nb15-dataset

[^9]: https://linkinghub.elsevier.com/retrieve/pii/S2210670721002808

[^10]: https://ro.ecu.edu.au/ecuworks2022-2026/552/

[^11]: https://www.emergentmind.com/topics/nf-cse-cic-ids2018

[^12]: https://ui.adsabs.harvard.edu/abs/2020IEEEA...8p5130A/abstract

[^13]: https://www.sciencedirect.com/science/article/abs/pii/S0167404817302353

[^14]: https://www.unb.ca/cic/datasets/ids-2017.html

[^15]: https://www.unb.ca/cic/datasets/nsl.html

[^16]: https://github.com/naviprem/ids-deep-learning/blob/master/datasets/UNSW-NB15.md

[^17]: http://www.scitepress.org/DigitalLibrary/Link.aspx?doi=10.5220/0006639801080116

[^18]: https://intrusion-detection.distrinet-research.be/WTMC2021/extended_doc.html

[^19]: https://dl.acm.org/doi/10.1007/978-3-031-31108-6_2

[^20]: https://www.semanticscholar.org/paper/71750d528b066250d7439ac1950bd31bcc16ce63

[^21]: https://www.yorku.ca/research/bccc/ucs-technical/cybersecurity-datasets-cds/large-scale-intrusion-detection-dataset-bccc-cse-cic-ids2018/

[^22]: https://ieeexplore.ieee.org/document/8888419/

[^23]: https://research.unsw.edu.au/projects/bot-iot-dataset

[^24]: https://www.kaggle.com/datasets/arnobbhowmik/ton-iot-network-dataset

[^25]: https://ieeexplore.ieee.org/document/9189760/

[^26]: https://researchdata.edu.au/new-generations-internet-toniot-datasets/1957406

[^27]: https://ro.ecu.edu.au/ecuworkspost2013/9883/

[^28]: https://www.nature.com/articles/s41598-026-37834-y

[^29]: https://www.unb.ca/cic/research/applications.html

[^30]: https://pure.uj.ac.za/en/publications/deep-learning-based-network-intrusion-detection-systems-a-systema/

[^31]: https://pypi.org/project/sb3-contrib/

[^32]: https://www.springerprofessional.de/en/faulty-use-of-the-cic-ids-2017-dataset-in-information-security-r/26573362

[^33]: https://ieeexplore.ieee.org/document/10729979/

[^34]: https://pubs.aip.org/aip/acp/article-lookup/doi/10.1063/5.0203394

[^35]: https://ieeexplore.ieee.org/document/5749879/

[^36]: https://www.semanticscholar.org/paper/50bf39d46bd273bb023e041537607f4ee8f7d587

[^37]: https://linkinghub.elsevier.com/retrieve/pii/S1877050920317804

[^38]: https://www.semanticscholar.org/paper/80320cf46e2110904ae1e48ebd11043fce982cab

[^39]: https://www.semanticscholar.org/paper/74a372c488622a70fec181bed702131dc49fe790

[^40]: https://actainformaticamalaysia.com/archives/AIM/2aim2022/2aim2022-55-61.pdf

[^41]: https://www.slideshare.net/slideshow/analysis-of-the-kdd-cup1999-datasets/76625551

[^42]: https://github.com/Nour-Moustafa/TON_IoT-Network-dataset

[^43]: https://www.iaras.org/home/caijels/hybrid-intrusion-detection-with-edge-iiotset-dataset

[^44]: https://ieeexplore.ieee.org/document/9343133/

[^45]: https://ieeexplore.ieee.org/document/9343084/

[^46]: http://www.aimspress.com/article/doi/10.3934/mbe.2022493

[^47]: http://www.aimspress.com/article/doi/10.3934/mbe.2023602

[^48]: https://arxiv.org/pdf/2306.02715.pdf

[^49]: https://www.mdpi.com/2224-2708/12/1/3/pdf?version=1672222140

[^50]: https://www.studocu.com/in/document/anna-university/introduction-to-iot/edge-iiotset-comprehensive-cybersecurity-dataset-for-iot-iiot-applications-in/149861909

[^51]: https://www.semanticscholar.org/paper/UGR'16:-A-new-dataset-for-the-evaluation-of-network-Maciá-Fernández-Camacho/fbaa2655dcf7b1e1c7aedf9ad6e10880df8ebd0d

[^52]: https://www.studocu.vn/vn/document/hutech-university-of-technology/tai-lieu-thiet-bi/edge-iiot-set-a-comprehensive-cybersecurity-dataset-for-ml-dl/139733273

[^53]: https://huggingface.co/datasets/JimXie/IIoTset

[^54]: https://ieeexplore.ieee.org/document/9343133

[^55]: https://ijeret.org/index.php/ijeret/article/view/478

