<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# CICIDS2017 Deep Dossier

## 1. What CICIDS2017 is

CICIDS2017 is a flow‑based intrusion detection evaluation dataset released by the Canadian Institute for Cybersecurity (CIC) to address perceived deficiencies of legacy IDS datasets such as KDD’99 and DARPA, particularly outdated traffic patterns, limited attack diversity, and missing features/metadata.[CICIDS2017-Web; Sharafaldin2018-ICISSP — evidence: strong; caveat: design claims come from the dataset authors; use: justify the dataset’s role as a modern benchmark][^1][^2]
It contains full packet captures (PCAPs) and derived labeled flow CSVs generated with CICFlowMeter, covering five days (Mon–Fri, 3–7 July 2017) of traffic in a small enterprise‑style network with about a dozen victim hosts and a separate attacker network.[CICIDS2017-Web; Sharafaldin2018-ICISSP — strong; caveat: the network is still a research lab; use: describe capture context and motivate flow‑based modeling][^1]

Benign background traffic is generated via the B‑Profile system, which profiles the behavior of 25 users interacting via HTTP/HTTPS, FTP, SSH, and email, then replays these patterns to approximate realistic enterprise usage.[CICIDS2017-Web; BProfile2016 — strong; caveat: “realism” is based on profiling limited users and services; use: support a nuanced, not over‑optimistic, realism claim][^3][^1]
On attack days, a scripted attacker (primarily a Kali machine) launches eight attack families (Brute Force, DoS, DDoS, Web Attack, Heartbleed, Infiltration, Botnet, PortScan) at specified time windows against selected victims, creating a mix of benign and malicious flows.[CICIDS2017-Web; Sharafaldin2018-ICISSP — strong; caveat: attack implementations are tool‑specific and some are flawed (e.g., DoS Hulk); use: describe attack coverage and later critique weaknesses][^4][^2][^1]

Subsequent analyses by Sharafaldin et al., Oyelakin et al., and others show that CICIDS2017 is heavily imbalanced, with certain attack types (e.g., DoS Hulk) dominating the malicious traffic and some families being rare.[Sharafaldin2019-Detailed; Oyelakin2023-Overview — moderate; caveat: exact counts depend on the CSV version and cleaning; use: motivate careful class handling and data‑efficiency experiments][^5][^6][^4]
Later work has also revealed non‑trivial issues in flow construction, feature computation, and labeling (e.g., CICFlowMeter bugs, mislabelled flows, duplicated and all‑zero features), and several “fixed” derivatives of CICIDS2017 have been proposed (WTMC‑2021, CRiSIS‑2022, NFStream‑based NFS‑2023‑{nTE,TE}).[Engelen2021-WTMC; Lanvin2023-Errors; Pekar2024-NFStream; Zhang2025-OpenSet — strong; caveat: fixes differ in scope; use: support a critical, evidence‑based limitations section][^7][^8][^9][^4]

***

## 2. Official facts

*(Use this table for the “official description” subsection; values that come from secondary analyses are marked accordingly.)*


| Fact | Value / description | Source (citation key) | Confidence |
| :-- | :-- | :-- | :-- |
| Dataset origin | Produced by Canadian Institute for Cybersecurity (UNB) | CICIDS2017-Web[CICIDS2017-Web][^1] | High (official site) |
| Purpose | Benchmark dataset for IDS/IPS evaluation with modern attacks and realistic benign traffic | CICIDS2017-Web; Sharafaldin2018-ICISSP[CICIDS2017-Web; Sharafaldin2018-ICISSP][^1][^2] | High |
| Capture period | 3–7 July 2017, Mon 9:00 to Fri 17:00 | CICIDS2017-Web[CICIDS2017-Web][^1] | High |
| Days | Monday: benign only; Tue–Fri: benign + attacks | CICIDS2017-Web[CICIDS2017-Web][^1] | High |
| Traffic formats | PCAPs, flow CSVs (GeneratedLabelledFlows.zip, MachineLearningCSV.zip) | CICIDS2017-Web[CICIDS2017-Web][^1] | High |
| Benign traffic generation | B‑Profile system modeling 25 user behaviors over HTTP(S), FTP, SSH, email | CICIDS2017-Web; BProfile2016[CICIDS2017-Web; BProfile2016][^1][^3] | High |
| Network topology | One “victim” LAN (~12 hosts: Windows, Ubuntu, macOS) behind firewall; separate attacker network (Kali + Windows) | CICIDS2017-Web[CICIDS2017-Web][^1] | High |
| Capture point | Mirror (SPAN) port on main switch, all traffic recorded to storage server | CICIDS2017-Web[CICIDS2017-Web][^1] | High |
| Attack families | Brute Force FTP/SSH, DoS (Slowloris, SlowHTTPTest, Hulk, GoldenEye), Heartbleed, Web Attacks (Brute, XSS, SQLi), Infiltration, Botnet ARES, PortScan, DDoS LOIT | CICIDS2017-Web; Sharafaldin2018-ICISSP[CICIDS2017-Web; Sharafaldin2018-ICISSP][^1][^2] | High |
| Label granularity | Flow‑level labels including benign and attack types | CICIDS2017-Web; Sharafaldin2018-ICISSP[CICIDS2017-Web; Sharafaldin2018-ICISSP][^1][^2] | High |
| Flow generator | CICFlowMeter (aka ISCXFlowMeter) | CICIDS2017-Web; Lashkari2017-CICFlowMeter[CICIDS2017-Web; Lashkari2017-CICFlowMeter][^1][^10] | High |
| Feature count | “More than 80” bidirectional flow features per flow; common CSVs show 78 numeric + label, plus sometimes Flow ID/Timestamp | CICIDS2017-Web; Lashkari2017-CICFlowMeter; Sharafaldin2019-Detailed[CICIDS2017-Web; Lashkari2017-CICFlowMeter; Sharafaldin2019-Detailed][^1][^10][^5] | High |
| File organization | CSVs and PCAPs per day; some public mirrors regroup by attack family | CICIDS2017-Web; secondary mirrors (Kaggle, GitHub)[CICIDS2017-Web; Kaggle-CICIDS2017][^1][^11] | High for official, low–moderate for mirrors |
| Class distribution | Highly imbalanced; DoS Hulk and some other classes dominate in number of flows; some families rare | Sharafaldin2019-Detailed; Oyelakin2023-Overview; Engelen2021-WTMC[Sharafaldin2019-Detailed; Oyelakin2023-Overview; Engelen2021-WTMC][^5][^6][^4] | Moderate (depends on version/cleaning) |
| Known CSV issues | NaN and Infinity in some packet length features; duplicate Fwd Header Length column; multiple all‑zero flag/bulk features | Zhang2025-OpenSet; Engelen2021-WTMC[Zhang2025-OpenSet; Engelen2021-WTMC][^7][^4] | High |
| CICFlowMeter bugs | TCP flag counts limited to {0,1}; incorrect PSH/URG flag counts; issues with flow direction under timeouts | Engelen2021-WTMC[Engelen2021-WTMC][^4] | High |
| Labelling issues | Attack time windows on website insufficient to reproduce labels; some flows mislabelled due to flow direction bug and timing windows | Engelen2021-WTMC; Lanvin2023-Errors[Engelen2021-WTMC; Lanvin2023-Errors][^4][^8] | High |
| Corrected variants | WTMC‑2021, CRiSIS‑2022, NFStream NFS‑2023‑{nTE,TE} provide re‑extracted and relabelled flows | Engelen2021-WTMC; Lanvin2023-Errors; Pekar2024-NFStream[Engelen2021-WTMC; Lanvin2023-Errors; Pekar2024-NFStream][^4][^8][^9] | High |


***

## 3. Attack categories and labels

*(This table is tuned to your binary PERMIT/BLOCK framing; Codex can compress it when drafting.)*


| Attack family | Description | Typical traffic behavior | Relevance to binary PERMIT/BLOCK | Caveats |
| :-- | :-- | :-- | :-- | :-- |
| Benign | Background user traffic from 25 profiled users (web browsing, email, SSH, FTP, etc.) generated by B‑Profile | Mix of HTTP(S), SMTP, IMAP, SSH, FTP flows with realistic inter‑arrival times and packet sizes | Ground truth negative class (PERMIT) and basis for FP analysis | Benign patterns limited to scripted profiles and small LAN; no streaming, P2P, or exotic apps[CICIDS2017-Web — strong; caveat: limited service diversity; use: justify that benign = “enterprise‑like but not exhaustive”][^1][^3] |
| Brute Force FTP-Patator | Automated password guessing against FTP service | Many short authentication attempts from attacker to FTP port; repeated failed login sequences | Positive (BLOCK) flows originate from attacker’s IP and target FTP server; simple signature‑like patterns | Tool‑dependent; in flow‑level representation, patterns may be easier than in real stealthy brute force (e.g., slower, distributed) [Sharafaldin2018-ICISSP — strong][^2][^1] |
| Brute Force SSH-Patator | Automated SSH password guessing | Similar to FTP-Patator but targeting SSH | Same as above | Same caveats as FTP; oversimplified attacker behavior [Sharafaldin2018-ICISSP — strong][^2] |
| DoS Slowloris / SlowHTTPTest | Application‑layer DoS keeping many HTTP connections partially open | Numerous HTTP flows with unusual timing: partial requests, long idle/active periods | BLOCK decisions correspond to flows tied to attack windows and victim web server | Apache configuration (KeepAliveTimeout, Timeout) weakens attacks, leading to non‑representative behavior and shortcut learning via timing patterns[Engelen2021-WTMC — strong][^4] |
| DoS Hulk | Intended HTTP flood with keep‑alive abuse | Many HTTP GETs with “Keep‑Alive” headers | BLOCK: flows labelled DoS Hulk | Implementation in CICIDS2017 is broken due to urllib2 overriding Connection header to “close”; attack largely ineffective and may behave like benign heavy browsing[Engelen2021-WTMC — strong; caveat: still widely used as DoS label; use: justify excluding or downweighting Hulk in serious evaluations][^4] |
| DoS GoldenEye | Correctly implemented HTTP keep‑alive abuse | Large numbers of HTTP requests with high keep‑alive seconds | BLOCK flows; more realistic DoS than Hulk | Apache KeepAliveTimeout=5 constrains connection duration and may create environment‑specific shortcuts[Engelen2021-WTMC — moderate][^4] |
| Heartbleed | Exploitation of OpenSSL heartbeat vulnerability | TLS heartbeat packets on port 444, abnormal payload sizes | BLOCK flows represent high‑impact but rare attack | Very constrained scenario (specific host/port); not all flows labelled malicious may be directly exploitable [CICIDS2017-Web — moderate][^1] |
| Web Attacks (BruteForce, XSS, SQLi) | HTTP application attacks against web server | Specific HTTP methods/URLs and payloads; often small number of flows | BLOCK flows reflect application‑level misuse | Narrow coverage of web attack types; manual scripts; payload not always visible at flow level [Sharafaldin2018-ICISSP — moderate][^2] |
| Infiltration | Compromised internal host performing scanning / malware download | Initial exploitation (Metasploit), internal scanning from victim, Dropbox malware download | BLOCK flows correspond to lateral movement and exfiltration patterns | Time windows and exact labeling non‑trivial; some post‑exploitation traffic may be labelled benign or vice versa[Engelen2021-WTMC — moderate][^4][^1] |
| Botnet ARES | Botnet C2 and malicious traffic from multiple infected hosts | Coordinated flows from bots to C2 and targets | BLOCK flows denote infected hosts; important for host‑level decisions | Botnet scenario relies on specific ARES implementation; C2 patterns limited to one family[CICIDS2017-Web — moderate][^1] |
| PortScan | Nmap and other scans toward victim | Many short flows to diverse ports on victim | BLOCK flows represent reconnaissance | Labeling uses coarse attack windows; some scans may be missed or mislabeled, and port‑related features are highly leakage‑prone[Engelen2021-WTMC; CICIDS2017-Web — strong][^4][^1] |
| DDoS LOIT | Distributed DoS from three Windows hosts toward web server | Many concurrent HTTP or TCP flows from multiple sources | BLOCK flows represent volumetric behavior in flow space | Limited number of attacking hosts; patterns may be easier than truly distributed attacks[CICIDS2017-Web — moderate][^1] |


***

## 4. Feature extraction and flow representation

**What a “flow” is.**
CICFlowMeter constructs bidirectional flows keyed on 5‑tuple (src IP, src port, dst IP, dst port, protocol) and direction, with timeouts (120 s) and termination upon seeing FIN flags.[Lashkari2017-CICFlowMeter; CICIDS2017-Web — strong; caveat: actual behavior deviates from TCP spec in CICFlowMeter versions used; use: accurately define flows in the thesis][^10][^4][^1]
A flow aggregates all packets in that connection segment, and the CSVs contain one row per flow with features and a label derived from attack time windows and host roles.[Sharafaldin2018-ICISSP; Engelen2021-WTMC — strong][^2][^4]

**Feature types.**
CICFlowMeter extracts >80 features, commonly grouped as: identifiers (Flow ID, Timestamp, Src/Dst IP, ports, protocol), basic counts (total packets/bytes, flow duration), per‑direction stats (Fwd/Bwd packet counts/bytes), packet length statistics (min/max/mean/std, for total, Fwd, Bwd), time features (inter‑arrival times, active/idle times), TCP flag counts, window sizes, and some derived rates (bytes/second, packets/second, bulk measures).[Lashkari2017-CICFlowMeter; CICIDS2017-Web — strong; caveat: exact list depends on CSV variant; use: justify the canonical feature vector for each flow][^7][^10][^1]

**Useful feature groups for learning.**
Empirical analyses (Sharafaldin et al., later benchmarking, and exploratory EDA) show that duration, packet counts, byte counts, packet length statistics, and inter‑arrival times are highly informative for differentiating benign vs. DoS/DDoS and brute force attacks.[Sharafaldin2019-Detailed; BenchmarkingCICIDS2017; Oyelakin2023-Overview — moderate; caveat: most analyses use supervised ML, not RL; use: motivate focusing on these feature groups for RL and RF baselines][^8][^6][^5]

**Dangerous or leakage‑prone features.**

Several studies explicitly recommend removing or carefully handling certain features:

- **Identifiers and routing fields**: Flow ID, Src IP, Dst IP, Src Port, Dst Port, protocol.
These directly encode scenario‑specific information (e.g., that a given IP is the Kali attacker or that a specific port only appears during attacks), which can lead to models that memorize the experimental setup rather than learning attack behavior.[Zhang2025-OpenSet; Engelen2021-WTMC; FaultyUseCICIDS2017 — strong; caveat: may still be needed in some operational settings; use: justify excluding them for fair benchmarking][^12][^4][^7]
- **Timestamp‑related fields**: Absolute time stamps and certain active/idle time features.
WTMC and follow‑up work show that CICFlowMeter incorrectly encoded absolute timestamps into some active/idle statistics in some versions, and dataset splits based purely on random rows can exploit experiment‑specific timing patterns.[Engelen2021-WTMC; Zhang2025-OpenSet — strong; caveat: original CICIDS2017 PCAP→CSV pipeline did not include all later bugs; use: treat time features carefully and prefer relative durations][^4][^7]
- **All‑zero features**: Bwd PSH Flags, Fwd/Bwd URG Flags, CWE Flags Count, Fwd/Bwd Avg Bytes/Bulk, Fwd/Bwd Avg Packets/Bulk, Fwd/Bwd Avg Bulk Rate.
These features are constant zero across all samples and thus useless; they should be dropped.[Zhang2025-OpenSet — strong; caveat: list may vary slightly per CSV version; use: safe feature pruning step][^7]
- **Duplicate columns**: Fwd Header Length appears twice with identical values in some CSVs due to a CICFlowMeter bug.[Engelen2021-WTMC; Zhang2025-OpenSet — strong; caveat: some mirrors already removed it; use: drop duplicate column and document it][^4][^7]

**Claim (leakage awareness).**
It is widely acknowledged in recent CICIDS2017 critiques that keeping IP addresses, ports, and time‑window‑dependent fields in the feature set, combined with random row‑wise splitting, leads to strong dataset‑specific shortcuts and over‑optimistic estimates of detection performance.[Engelen2021-WTMC; Lanvin2023-Errors; FaultyUseCICIDS2017 — evidence: strong; caveat: the degree of leakage depends on the specific split and task; use: argue for dropping such features and using strict splits].[^8][^12][^4]

***

## 5. Known limitations

*(Focus on limitations that matter for your RL defender and evaluation.)*


| Limitation | Description | Why it matters | How to mitigate | Citation |
| :-- | :-- | :-- | :-- | :-- |
| Small, lab‑scale topology | Single small LAN with 12 victim hosts and one main firewall; attacker network with few machines | Models can memorize host‑specific patterns; not representative of large enterprise networks | Remove IP/port IDs; use flow‑intrinsic stats; treat CICIDS2017 as a controlled benchmark, not a proxy for Internet‑scale traffic | CICIDS2017-Web; Engelen2021-WTMC[CICIDS2017-Web; Engelen2021-WTMC][^1][^4] |
| Scripted benign traffic | Benign behavior generated by B‑Profile with 25 user profiles and limited protocols | Benign class diversity is limited; may not include modern apps (video, P2P, cloud APIs) | Explicitly state this; perform external validation on your lab flows; consider combining with other datasets in future work | CICIDS2017-Web; DatasetSurvey2025[CICIDS2017-Web; DatasetSurvey2025][^1][^13] |
| Imbalanced class distribution | Certain attacks (DoS Hulk, PortScan, DDoS) dominate; some classes are rare | Models can achieve high accuracy by focusing on frequent attacks; minority classes under‑represented | Use class‑wise metrics (per‑class recall/F1), cost‑sensitive losses, and downsample dominant attack types in some experiments | Sharafaldin2019-Detailed; Oyelakin2023-Overview[Sharafaldin2019-Detailed; Oyelakin2023-Overview][^5][^6] |
| CICFlowMeter bugs (flags, flow direction) | TCP flag counts incorrect; flow direction handling under timeouts leads to mislabelled flows | Labels and flag features are partially unreliable; models may learn buggy patterns | Consider using corrected datasets (WTMC‑2021/NFS‑2023) or at least drop flag count features and acknowledge label noise | Engelen2021-WTMC; Lanvin2023-Errors[Engelen2021-WTMC; Lanvin2023-Errors][^4][^8] |
| NaN and Infinity values | Bwd Packet Length Max/Min contain NaN and Infinity for some flows | Many algorithms (and RL implementations) cannot handle NaN/Inf; naive dropping leads to subtle sample selection bias | Impute or cap values in a documented way (e.g., per‑class mean/max); log counts before/after cleaning | Zhang2025-OpenSet[Zhang2025-OpenSet][^7] |
| Duplicate and all‑zero features | Duplicate Fwd Header Length; multiple all‑zero flag/bulk features | Inflates dimensionality; can lead to ill‑conditioned scaling; wastes capacity | Remove duplicate and all‑zero features in preprocessing | Zhang2025-OpenSet; Engelen2021-WTMC[Zhang2025-OpenSet; Engelen2021-WTMC][^7][^4] |
| Label timing uncertainty | Website attack windows not sufficient to reproduce exact labels; some flows mislabelled | Hard to reconstruct labels from PCAPs; minor label noise across flow boundaries | Use official CSVs or corrected published variants; if you re‑extract, use published attack windows and scripts (WTMC code, NFStream) | Engelen2021-WTMC; Pekar2024-NFStream[Engelen2021-WTMC; Pekar2024-NFStream][^4][^9] |
| Flawed attack implementations | DoS Hulk ineffective; GoldenEye and slow attacks constrained by Apache config; some attacks more artifact‑driven than realistic | Models may learn artifacts (KeepAliveTimeout behaviors, misimplemented tools) rather than generic attack patterns | De‑emphasize Hulk in evaluations; analyze per‑attack performance; avoid over‑claiming general DoS detection | Engelen2021-WTMC[Engelen2021-WTMC][^4] |
| Evaluation misuse in literature | Many papers use random row‑wise splits, keep IP/port features, and report near‑perfect accuracy | Leads to inflated, non‑deployable performance claims | Use strict splits (by day/scenario/host), remove leakage‑prone features, report multiple seeds and robust metrics | Lanvin2023-Errors; FaultyUseCICIDS2017; Evaluation critiques[Lanvin2023-Errors; FaultyUseCICIDS2017][^8][^12] |
| Dataset variants and incompatibility | Multiple “fixed” versions (WTMC‑2021, CRiSIS‑2022, NFS‑2023) exist with different flows and features | Difficult to compare results across works without specifying exact variant | Clearly state which release and preprocessing pipeline are used; document hashes or source URLs | Engelen2021-WTMC; Pekar2024-NFStream[Engelen2021-WTMC; Pekar2024-NFStream][^4][^9] |


***

## 6. Preprocessing checklist for my thesis

Below is a concrete checklist tailored to your RL‑defender pipeline.

**Columns and feature selection**

- **Load a well‑defined CSV version** (e.g., official MachineLearningCSV.zip or a corrected WTMC/NFS version) and record its source URL and checksum.[Engelen2021-WTMC; Pekar2024-NFStream — strong; caveat: different versions differ in exact schema; use: guarantee reproducibility][^9][^4]
- **Drop leakage‑prone identifiers:** Flow ID, Src IP, Dst IP, Src Port, Dst Port, protocol, if present.[Engelen2021-WTMC; Zhang2025-OpenSet — strong; caveat: if you want to study IP‑aware policies later, treat as a separate experiment; use: main evaluation avoids trivial shortcuts][^4][^7]
- **Remove all‑zero and duplicate features** as identified in high‑quality analyses (duplicate Fwd Header Length; the 10 all‑zero flag/bulk features).[Zhang2025-OpenSet — strong; use: justify feature pruning step][^7]

**Label mapping**

- **Normalize label strings** (trim whitespace, unify capitalization) and map known attack labels to a consistent taxonomy (e.g., FTP-Patator → BruteForce, Web Attack \* → WebAttack, etc.) while keeping the original string in a metadata field for traceability.[CICIDS2017-Web; Sharafaldin2019-Detailed — moderate; caveat: some mirrors already normalized; use: robust label handling][^5][^1]
- **Create a binary label** `y_binary ∈ {0,1}` where 0 = BENIGN, 1 = any attack label, and store both multi‑class and binary labels to allow later analyses.[Sharafaldin2019-Detailed — moderate; use: facilitates binary PERMIT/BLOCK RL and multi‑class ablations][^5]

**NaN/Inf handling**

- **Detect NaN/Inf** specifically in `Bwd Packet Length Max`, `Bwd Packet Length Min`, and any other numeric feature.[Zhang2025-OpenSet — strong][^7]
- **Impute or cap values**: e.g., replace NaN with per‑feature mean on benign flows and Inf with the maximum finite value; log counts of affected rows before/after.[Zhang2025-OpenSet — moderate; caveat: alternatives (deletion, per‑class imputation) are possible; use: ensures SB3 QRDQN receives finite observations][^7]

**Scaling**

- **Standardize or normalize numeric features** (e.g., z‑score or robust scaling) based on training data only, then apply the same transformation to validation/test and lab traffic.[ML‑IDS best practices — moderate; caveat: RL sometimes works unscaled but distributional stability is better with normalized inputs; use: justify scaling choice in methodology][^14][^15]

**Train/test split discipline**

- **Decide on evaluation unit** (e.g., day‑based or scenario‑based). For a conservative split, use: Train = Monday + part of Tuesday/Wednesday benign and attacks; Validation = separate subset; Test = remaining days or attack scenarios, ensuring no overlap in flows.[Engelen2021-WTMC; Lanvin2023-Errors — moderate; caveat: there is no canonical split; use: describe and justify your own strict split][^8][^4]
- **Avoid random row‑wise splits**; if you use any random split for data‑efficiency curves, restrict it to the training set while keeping a fixed temporal hold‑out test set.[FaultyUseCICIDS2017; CrossDomain evaluations — strong; use: avoid inflated performance claims][^16][^12]

**Reproducibility logging**

- **Log preprocessing code and configuration** (Git commit, script paths, feature list, scaling parameters, dataset version).
- **Store train/val/test indices** (e.g., as arrays of row IDs) so experiments with different algorithms (RF, QRDQN) share the same split.[ReproBaseline-ML-IDS — moderate; use: show methodological rigor][^14]
- **Fix RNG seeds** for train/val splits, scaling, and model training; record them in the thesis appendix and repository.[SB3/Gym docs — moderate; use: supports multi‑seed experiments][^17][^18]

***

## 7. Evaluation protocol recommendation

Given your project constraints and the literature, a reasonable protocol is:

1. **Internal public benchmark (CICIDS2017 only).**
    - Use a **strict split**: e.g., train on Monday + part of Tue–Thu, validate on a small slice of these days, and test on flows from held‑out time intervals or full unseen attack windows (e.g., some afternoon attacks and Friday scenarios).[Engelen2021-WTMC; Lanvin2023-Errors — moderate; use: define a deployment‑like test set][^8][^4]
    - Apply the same split to RF baseline and QRDQN environment to enable fair comparison.
2. **Fixed holdout and multiple seeds.**
    - Fix the train/val/test partition and **run multiple seeds** for RF (if stochastic) and QRDQN; report means and standard deviations of binary metrics (accuracy, precision, recall, F1, AUROC, FPR, FNR).[ReproBaseline-ML-IDS; DRL evaluation norms — moderate][^19][^14]
3. **Data‑efficiency experiments.**
    - On the training set only, subsample flows (e.g., 1%, 5%, 10%, 25%, 50%, 100%) and train RF and QRDQN from scratch to obtain performance vs. data curves.[Sharafaldin2019-Detailed; Pekar2024-NFStream — moderate; use: show how RL behaves under limited labeled data][^9][^5]
4. **False positive / false negative analysis.**
    - For the binary task, compute FPR/FNR overall and per attack family where possible; relate these to cost‑sensitive models from the literature (cost of blocking benign vs. missing attacks).[FP/FN‑CostModel; CSE-IDS — moderate; use: motivate reward shaping][^20][^21]
5. **External lab validation (Phase 2).**
    - Train RF and QRDQN exclusively on CICIDS2017 training split.
    - Evaluate **offline** on flow features extracted from your private lab PCAPs using the same CICFlowMeter configuration or an equivalent tool (e.g., NFStream‑based flow extraction).[Farrukh2022-PayloadByte; Pekar2024-NFStream — moderate; caveat: you must ensure feature compatibility; use: support generalization claims beyond CICIDS2017][^22][^9]
    - Report performance and qualitative error analysis (e.g., types of benign flows misclassified as attacks and vice versa), stressing any domain shift.
6. **Optional: compare original vs. corrected CICIDS2017.**
    - If time allows, run a small subset of experiments on a corrected derivative (WTMC/NFS‑2023) to see whether QRDQN and RF performance is robust to dataset “quality”.[Pekar2024-NFStream — moderate][^9]

**Claim (random splits).**
Literature on CICIDS2017 explicitly warns that random row‑wise splits with IP/port features preserved lead to unrealistically high performance, and better‑controlled studies argue for scenario‑ or time‑based splits.[Lanvin2023-Errors; FaultyUseCICIDS2017 — strong; caveat: some earlier works did not have access to corrected datasets; use: strongly justify not using random splits as main evaluation].[^12][^8]

***

## 8. Thesis-ready wording (4 paragraphs, content-ready for Spanish drafting)

*(These are conceptual paragraphs in English; Codex should translate and adapt tone.)*

1. **What CICIDS2017 is.**
CICIDS2017 is an intrusion detection evaluation dataset released by the Canadian Institute for Cybersecurity that combines full packet captures and labelled flow records for five days of traffic in a small enterprise‑style network.[CICIDS2017-Web; Sharafaldin2018-ICISSP — strong; caveat: lab‑scale, not Internet‑scale; use: definitional paragraph][^2][^1]
Benign traffic is generated using the B‑Profile system, which models the behavior of 25 users interacting over HTTP(S), FTP, SSH, and email, while several families of attacks—brute force against FTP and SSH, multiple application‑layer DoS and DDoS tools, Heartbleed, web attacks, infiltration scenarios, botnet C2 and port scans—are executed at controlled time windows against selected victim hosts.[CICIDS2017-Web; Lashkari2017-CICFlowMeter — strong; caveat: attack tools and services represent a limited subset of real threats; use: dataset narrative][^10][^1]
2. **Why it is used.**
Compared to legacy datasets such as KDD’99 and DARPA, CICIDS2017 offers more recent traffic, richer flow‑level features, and explicit documentation of network topology, traffic generation, and attack scenarios, and has therefore become a de facto benchmark for machine‑ and deep‑learning‑based IDS research.[Sharafaldin2018-ICISSP; DatasetSurvey2025 — strong; caveat: other modern datasets (UNSW‑NB15, CSE‑CIC‑IDS2018, CIC‑DDoS2019) also exist; use: justification paragraph][^13][^2]
Its flow CSVs, generated with CICFlowMeter, provide more than 80 features per bidirectional flow, including duration, packet and byte counts, packet length statistics and timing information, which are well suited to flow‑based detectors such as the Random Forest baselines and RL‑based defenders considered in this thesis.[Lashkari2017-CICFlowMeter; Sharafaldin2019-Detailed — strong; caveat: some features are redundant or flawed; use: connect CICIDS2017 to your feature vector and RL environment][^10][^5]
3. **Limitations.**
At the same time, recent independent analyses reveal important limitations of CICIDS2017, including bugs in the CICFlowMeter tool that affect TCP flag counts and flow direction, duplicate and all‑zero features, NaN and Infinity values in packet length statistics, and inaccuracies in flow labels due to coarse attack time windows and flow construction logic.[Engelen2021-WTMC; Lanvin2023-Errors; Zhang2025-OpenSet — strong; caveat: severity depends on the specific CSV release; use: limitation paragraph][^4][^8][^7]
Moreover, the dataset’s small lab‑scale topology and scripted benign behavior mean that models can easily exploit IP addresses, ports, and environment‑specific timing patterns if these are left in the feature set and random row‑wise splits are used, leading to over‑optimistic performance that does not reflect deployment settings.[FaultyUseCICIDS2017; Pekar2024-NFStream — strong; caveat: some earlier works lacked awareness of these issues; use: motivate strict preprocessing and splitting in your thesis][^12][^9]
4. **Why external validation is necessary.**
Given these limitations, this thesis treats CICIDS2017 primarily as a controlled benchmark to compare algorithms under carefully designed preprocessing and splitting protocols, rather than as a faithful representation of all real network environments.[Lanvin2023-Errors; Evaluation critiques — moderate; caveat: still widely used in literature; use: frame your evaluation scope][^16][^8]
To assess whether the proposed RL‑based defender actually learns patterns that transfer beyond this specific lab setup, the thesis includes a second evaluation phase in which models trained on CICIDS2017 are run offline on flows extracted from independent traffic captured in a separate laboratory environment, providing an external validation step that complements the internal benchmark and reduces the risk of overstating generalization performance.[Pekar2024-NFStream; Farrukh2022-PayloadByte — moderate; caveat: your lab traffic will still be limited; use: justify Phase 2][^22][^9]

***

## 9. BibTeX and citation keys

*(Keys aligned with those used above; fill missing fields from PDFs when you have institutional access.)*

```bibtex
@inproceedings{Sharafaldin2018-ICISSP,
  author    = {Iman Sharafaldin and Arash Habibi Lashkari and Ali A. Ghorbani},
  title     = {Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization},
  booktitle = {Proceedings of the 4th International Conference on Information Systems Security and Privacy (ICISSP)},
  year      = {2018},
  doi       = {10.5220/0006639801080116}
}

@incollection{Sharafaldin2019-Detailed,
  author    = {Iman Sharafaldin and Arash Habibi Lashkari and Ali A. Ghorbani},
  title     = {A Detailed Analysis of the CICIDS2017 Data Set},
  booktitle = {Communications in Computer and Information Science},
  year      = {2019},
  doi       = {10.1007/978-3-030-25109-3_9}
}

@misc{CICIDS2017-Web,
  author       = {{Canadian Institute for Cybersecurity}},
  title        = {Intrusion Detection Evaluation Dataset (CIC-IDS2017)},
  howpublished = {\url{https://www.unb.ca/cic/datasets/ids-2017.html}},
  note         = {Accessed 2026-05-14}
}

@inproceedings{Lashkari2017-CICFlowMeter,
  author    = {A. Habibi Lashkari and others},
  title     = {Characterization of Tor Traffic using Time based Features},
  booktitle = {Proceedings of ICISSP},
  year      = {2017}
}

@article{Engelen2021-WTMC,
  author  = {Gints Engelen and Pieter Maene and Wouter Joosen and others},
  title   = {Troubleshooting an Intrusion Detection Dataset: the CICIDS2017 Case Study},
  journal = {IEEE Security and Privacy Workshops (WTMC)},
  year    = {2021},
  doi     = {10.1109/SPW53761.2021.00009}
}

@incollection{Lanvin2023-Errors,
  author    = {Maxime Lanvin and Pierre-Fran{\c{c}}ois Gimenez and Yufei Han and Fr{\'e}d{\'e}ric Majorczyk and Ludovic M{\'e} and {\'E}ric Totel},
  title     = {Errors in the CICIDS2017 Dataset and the Significant Differences in Detection Performances It Makes},
  booktitle = {Lecture Notes in Computer Science},
  year      = {2023},
  doi       = {10.1007/978-3-031-31108-6_2}
}

@article{Oyelakin2023-Overview,
  author  = {A. Oyelakin and A. O. Ameen and T. S. Ogundele and others},
  title   = {Overview and Exploratory Analyses of CICIDS 2017 Intrusion Detection Dataset},
  journal = {Journal of Systems Engineering and Information Technology},
  volume  = {2},
  number  = {2},
  pages   = {45--52},
  year    = {2023},
  doi     = {10.29207/joseit.v2i2.5411}
}

@article{Zhang2025-OpenSet,
  author  = {Zhang, X. and others},
  title   = {Unknown intrusion traffic detection method based on unsupervised learning and open-set recognition},
  journal = {Scientific Reports},
  year    = {2025},
  doi     = {10.1038/s41598-025-01084-1}
}

@article{Pekar2024-NFStream,
  author  = {Adrian Pekar and Richard Jozsa},
  title   = {Evaluating ML-based Anomaly Detection Across Datasets of Varied Integrity: A Case Study},
  journal = {Computer Networks},
  year    = {2024},
  doi     = {10.1016/j.comnet.2024.110617},
  eprint  = {arXiv:2401.16843}
}

@article{Farrukh2022-PayloadByte,
  author  = {Yasir Ali Farrukh and Irfan Khan and Syed Wali and David Bierbrauer and Nathaniel Bastian},
  title   = {Payload-Byte: A Tool for Extracting and Labeling Packet Capture Files of Modern Network Intrusion Detection Datasets},
  journal = {TechRxiv Preprint},
  year    = {2022},
  doi     = {10.36227/techrxiv.20714221.v1}
}

@article{DatasetSurvey2025,
  author  = {B{\"o}nninghausen, P. U. and others},
  title   = {Network Intrusion Datasets: A Survey, Limitations, and Recommendations},
  journal = {Computers \& Security},
  year    = {2025}
}

@article{FaultyUseCICIDS2017,
  author  = {Author(s) omitted here for brevity},
  title   = {Faulty use of the CIC-IDS 2017 dataset in information security research},
  journal = {Journal/Publisher as per DOI 10.1007/978-3-XXX},
  year    = {2023},
  note    = {See SpringerProfessional article}
}
```

*(You can refine BibTeX once you have full metadata; keys should remain stable.)*

***

## 10. Codex handoff

**Goal:** Use this dossier to draft four Spanish subsections:

1. **“Dataset CICIDS2017”**
    - Describe origin, purpose, capture period, network topology, benign traffic generation (B‑Profile), and attack families.
    - Cite: `CICIDS2017-Web`, `Sharafaldin2018-ICISSP`, `Sharafaldin2019-Detailed`, `Lashkari2017-CICFlowMeter`.
    - Emphasize that it is a **flujo bidireccional** dataset with >80 características por flujo.
    - Do **not** claim it is fully realistic; mention explicitly that it is a **entorno de laboratorio controlado**.
2. **“Preprocesamiento”**
    - Follow the checklist in section 6.
    - Explicitly state:
        - Removal of IPs, puertos, Flow ID, protocolo and timestamps likely to induce leakage.
        - Eliminación de columnas duplicadas y de características con todo ceros (use list from `Zhang2025-OpenSet`).
        - Manejo de valores `NaN` e `Infinity` en longitudes de paquetes con imputación documentada.
        - Escalado de características numéricas con parámetros calculados sobre el conjunto de entrenamiento.
        - Estrategia de división entrenamiento/validación/prueba estricta basada en días/ventanas de ataque, **no** en filas aleatorias.
    - Cite: `Engelen2021-WTMC`, `Lanvin2023-Errors`, `Zhang2025-OpenSet`, `Pekar2024-NFStream`, `ReproBaseline-ML-IDS`.
3. **“Limitaciones del dataset”**
    - Use the table in section 5.
    - For each limitation, include **descripción**, **por qué importa**, and **cómo se mitiga en esta tesis**.
    - Key limitations to mention: tamaño reducido de la red, tráfico benigno sintético, desbalance de clases, errores de CICFlowMeter, ruido de etiquetas, implementaciones defectuosas de algunos ataques (especialmente DoS Hulk), problemas de evaluación en la literatura.
    - Cite primarily: `Engelen2021-WTMC`, `Lanvin2023-Errors`, `Oyelakin2023-Overview`, `Zhang2025-OpenSet`, `DatasetSurvey2025`, `FaultyUseCICIDS2017`.
4. **“Justificación de validación externa”**
    - Argue that, due to the above limitations and known misuse in previous works, CICIDS2017 alone is insufficient to claim generalization.
    - Explain that the thesis includes **Fase 2: inferencia offline** sobre flujos extraídos de tráfico de laboratorio independiente, manteniendo el mismo vector de características.
    - Make clear that Phase 2 is **validación externa**, not online blocking, and is used to see whether patrones aprendidos en CICIDS2017 se trasladan a otro entorno.
    - Cite: `Pekar2024-NFStream` (importance of dataset integrity), `Farrukh2022-PayloadByte` (PCAP↦flow pipelines), and cross‑dataset generalization work (`CrossDomain2023`, `DatasetSurvey2025`).

**Important drafting rules for Codex:**

- Always attach citation keys and evidence strength when making strong claims about CICIDS2017’s properties or limitations.
- Clearly distinguish between:
    - **Official claims** (Sharafaldin et al., CIC website)
    - **Independent analyses** (Engelen/WTMC, Lanvin/Errors, Zhang/Open‑set, Pekar/NFStream, Oyelakin/EDA)
    - **Secondary/implementation sources** (Kaggle, GitHub repos) — mark these as secundarios and avoid using them as the only support for any important claim.
- Avoid quoting dataset numbers (e.g., exact number of flows per class) unless you compute them from the exact CSV version used in the thesis; if you include counts, mention that they are **calculados en esta tesis**.
- When describing evaluation problems in prior literature, speak in general terms (e.g., “varios trabajos anteriores”) and base critiques on `FaultyUseCICIDS2017` and `Lanvin2023-Errors`, not on individual papers unless they are clearly documented.

With this, Codex has a precise map of what CICIDS2017 is, how you are preprocessing it, what its limitations are, and why you add external lab validation on top.
<span style="display:none">[^23][^24][^25][^26][^27][^28][^29][^30][^31][^32][^33][^34][^35][^36][^37][^38][^39][^40][^41][^42][^43][^44][^45][^46][^47][^48][^49][^50][^51][^52][^53]</span>

<div align="center">⁂</div>

[^1]: https://www.unb.ca/cic/datasets/ids-2017.html

[^2]: http://www.scitepress.org/DigitalLibrary/Link.aspx?doi=10.5220/0006639801080116

[^3]: https://data.mendeley.com/datasets/nrxz4sfj73

[^4]: https://intrusion-detection.distrinet-research.be/WTMC2021/extended_doc.html

[^5]: https://www.semanticscholar.org/paper/71750d528b066250d7439ac1950bd31bcc16ce63

[^6]: https://www.mendeley.com/catalogue/1ef6eeea-6e78-3e7a-a6e3-2385fe0033b0/

[^7]: https://www.nature.com/articles/s41598-025-01084-1

[^8]: https://dl.acm.org/doi/10.1007/978-3-031-31108-6_2

[^9]: https://arxiv.org/abs/2401.16843

[^10]: https://www.unb.ca/cic/research/applications.html

[^11]: https://www.kaggle.com/datasets/bertvankeulen/cicids-2017

[^12]: https://www.springerprofessional.de/en/faulty-use-of-the-cic-ids-2017-dataset-in-information-security-r/26573362

[^13]: https://www.sciencedirect.com/science/article/abs/pii/S0167404825001993

[^14]: https://ojs.sciencesforce.com/index.php/smij/article/view/268

[^15]: https://ieeexplore.ieee.org/document/11412845/

[^16]: https://www.sciencedirect.com/science/article/abs/pii/S0045790623001167

[^17]: https://gymnasium.farama.org/index.html

[^18]: https://pypi.org/project/sb3-contrib/

[^19]: https://arxiv.org/abs/1710.10044

[^20]: https://www.scirp.org/journal/paperinformation?paperid=43038

[^21]: https://www.sciencedirect.com/science/article/abs/pii/S0167404821003230

[^22]: https://zenodo.org/records/7258579

[^23]: https://jcoms.fesb.unist.hr/10.24138/jcomss-2024-0064/

[^24]: https://www.jait.us/show-254-1684-1.html

[^25]: https://ieeexplore.ieee.org/document/11233429/

[^26]: https://ieeexplore.ieee.org/document/11233343/

[^27]: https://fcc08321-8158-469b-b54d-f591e0bd3df4.filesusr.com/ugd/185b0a_b6d0c74a7ffd4235837cf04047d40750.pdf

[^28]: https://ieeexplore.ieee.org/document/11250555/

[^29]: https://ieeexplore.ieee.org/document/11102586/

[^30]: https://www.nature.com/articles/s41598-025-85248-z

[^31]: https://github.com/devarshpatel1506/Data-Analysis-for-Network-Traffic-Analysis

[^32]: https://fugumt.com/fugumt/paper_check/2506.19877v1_enmode

[^33]: http://irjaes.com/wp-content/uploads/2020/10/IRJAES-V5N2P184Y20.pdf

[^34]: https://www.techscience.com/cmc/v73n3/49109/html

[^35]: https://rpubs.com/mptrossbach/CICIDS2017

[^36]: https://www.academia.edu/123203184/Overview_and_Exploratory_Analyses_of_CICIDS2017_Intrusion_Detection_Dataset

[^37]: https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset

[^38]: https://www.kaggle.com/code/ericanacletoribeiro/cicids2017-comprehensive-data-processing-for-ml

[^39]: https://www.aanda.org/10.1051/0004-6361/202453111

[^40]: https://www.aanda.org/10.1051/0004-6361/202450853

[^41]: https://www.semanticscholar.org/paper/30a2250a10a9a7a14d1237e57ddf4a8a0474ad35

[^42]: https://academic.oup.com/mnras/article/490/2/2284/5579033

[^43]: https://www.acpjournals.org/doi/10.7326/0003-4819-150-11-200906020-00006

[^44]: https://www.semanticscholar.org/paper/06fb1ec937a0c649f4a12eefe1e23708552d04fc

[^45]: https://www.semanticscholar.org/paper/0a6c7466ee8474d6ad987e2c1a359c0c2496af14

[^46]: http://peer.asee.org/26907

[^47]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11794860/

[^48]: https://github.com/noushinpervez/Intrusion-Detection-CICIDS2017

[^49]: https://github.com/GintsEngelen/WTMC2021-Code/blob/main/README.md

[^50]: http://ijns.jalaxy.com.tw/contents/ijns-v23-n6/ijns-2021-v23-n6-p985-996.pdf

[^51]: https://www.scribd.com/document/982825149/2506-19877v2

[^52]: https://www.scribd.com/document/741148506/Overview-and-Exploratory-Analyses-of-CICIDS-2017-I

[^53]: https://github.com/GintsEngelen/WTMC2021-Code

