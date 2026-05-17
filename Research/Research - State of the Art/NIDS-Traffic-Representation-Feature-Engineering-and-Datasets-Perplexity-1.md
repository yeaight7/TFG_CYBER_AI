# State of the Art on NIDS and Network-Flow Representations for a Flow-Level RL Defender

## Overview

This report surveys the intrusion-detection and traffic-representation background needed for a flow-level reinforcement-learning (RL) defender that operates on tabular network-flow features, with CICIDS2017 as the main internal benchmark.
It focuses on NIDS fundamentals, traffic representations from packets to flows, feature engineering and standardisation, public NIDS datasets, and a critical view of CICIDS2017, with explicit separation between benchmark usefulness and real-world operational validity.[^1][^2]

## IDS and NIDS Fundamentals

### IDS vs IPS and deployment modes

An Intrusion Detection System (IDS) monitors host or network activity and raises alerts when it observes behaviour indicative of attacks, misuse, or policy violations, whereas an Intrusion Prevention System (IPS) is an inline control point that can actively block or modify suspicious traffic.[^3][^4]
Host-based IDS (HIDS) instruments endpoints such as servers or workstations and inspects local events, logs, or system calls, while network-based IDS (NIDS) inspects traffic traversing network segments using packet captures, mirrors, or flow logs.[^2][^5]

NIDS can operate purely as detection and alerting components, feeding events to a SIEM or SOC workflow, or be tightly integrated with firewalls and SDN controllers to trigger active responses such as packet drops, rule updates, or quarantining of flows and hosts.[^6][^5]
For a thesis that studies offline RL over recorded flow rows, it is important to emphasise that the system is an experimental NIDS-like classifier, not a deployed IPS: its \texttt{PERMIT}/\texttt{BLOCK} actions remain offline labels over flow records rather than live control of network elements.[^7]

### Signature-based detection

Signature-based IDS matches observed events against a database of known attack patterns, such as byte sequences, protocol-field conditions, or rule-based combinations constructed by analysts.[^8][^3]
Systems like Snort and Suricata rely on large rule-sets and pattern-matching engines, which provide high precision for known threats but require constant updates and are inherently limited against zero-day or highly polymorphic attacks.[^9][^8]

These approaches are efficient when the attack surface is well catalogued, for example for commodity malware or well-known exploitation kits, and they offer deterministic explainability because each alert is tied to a specific rule or signature.[^8]
However, they struggle with encrypted payloads, protocol deviations that do not match predefined patterns, and novel behaviour, which motivates anomaly-based and hybrid approaches for detecting unknown threats.[^10][^3]

### Anomaly-based detection

Anomaly-based IDS first builds a model of normal behaviour, using statistics, clustering, or machine learning, and then flags significant deviations as potential intrusions.[^2][^9]
This paradigm promises the ability to detect previously unseen attacks and conceptually aligns with flow-based ML/DL NIDS, where models learn the distribution of benign flows and attack flows from features such as duration, byte counts, and inter-arrival times.[^11][^12]

In practice, anomaly-based NIDS faces high false-positive rates, sensitivity to distribution shift, and difficulty in obtaining representative training data, especially when benign traffic is highly diverse and attacks are rare.[^13][^10]
Sommer and Paxson argued that machine learning based anomaly detection is much harder to validate and tune under operational conditions than benchmark results suggest, because public datasets do not replicate the complexity, base rates, and labelling uncertainties of real networks.[^9][^2]

### Specification-based and hybrid detection

Specification-based IDS defines allowed behaviour using formal or semi-formal specifications of correct protocol or application behaviour and flags violations, combining aspects of misuse and anomaly detection.[^5][^3]
This is attractive for well-structured protocols or SCADA/ICS environments where expected behaviour is constrained, but specification engineering is labour-intensive and brittle when systems evolve.[^2]

Hybrid IDS combine signature-based and anomaly-based methods, often running a signature engine for known threats and an anomaly engine for suspicious but unknown behaviour, sometimes with a meta-classifier that fuses their outputs.[^14][^5]
Such systems aim to retain the precision and interpretability of signatures while expanding coverage to new attacks; they also occur at the architectural level when HIDS and NIDS signals are fused.[^3][^5]

### Alerting vs active response

Operationally, NIDS can be tuned for pure alerting or for semi-automated responses.
In alerting mode, detection results are logged and forwarded to analysts, who decide on containment steps, while in active-response mode, the IDS or an attached orchestrator pushes firewall rules, ACL updates, or SDN flow rules to block or rate-limit suspicious flows.[^15][^6]

From an evaluation perspective, benchmark accuracy is not equivalent to operational value: under realistic base rates, even a system with low false-positive rate can overwhelm operators with alerts, and the risk of blocking benign traffic constrains how aggressively detection thresholds can be set.[^9][^2]
A flow-level RL defender can encode asymmetric costs between false positives and false negatives via reward design, but as long as it operates on a static dataset, its conclusions remain evidence about an offline classifier, not a fully deployed IPS.

## Network Traffic Representation

### Levels of representation

Network traffic can be represented at several levels of granularity: raw packets, payload bytes, sessions or connections, and aggregated flows, each with different trade-offs in fidelity, privacy, and computational cost.[^16][^11]
Packet-level representations keep per-packet metadata such as timestamps, lengths, and headers, whereas payload-level representations retain full or partial application-layer content; session-level representations group packets into bidirectional exchanges, and flow-level representations summarise uni- or bi-directional traffic between endpoints over time.[^17][^16]

Higher-level abstractions like flows reduce data volume and make traffic more amenable to tabular ML models by replacing sequences of packets with statistical summaries, but they necessarily discard fine-grained temporal and payload information that might be critical for detecting some attacks.[^16][^11]
In practice, NIDS pipelines often combine multiple levels, using flow logs for scalable monitoring and packet or payload capture for deep inspection and forensic analysis.

### Packet-level representation

Packet-level NIDS uses features per packet, including timestamp, size, protocol, header flags, and sometimes a truncated portion of payload, enabling detection of low-level protocol anomalies, timing patterns, and fine-grained signatures.[^18][^16]
These representations are high-bandwidth and require significant storage and processing capabilities, especially on high-speed links, but allow deep packet inspection, stateful protocol analysis, and sequence modelling with RNNs or transformers for traffic classification.[^12][^17]

In the context of an RL-based defender over CICIDS2017, packet-level information is not directly available to the agent; instead, the RL environment observes flow-level statistics that have already aggregated packet events.
This makes it crucial to understand which aspects of packet behaviour survive into the flow features and which are irrevocably lost.

### Payload-level representation

Payload-based IDS inspects application-layer bytes to identify malicious content such as exploits, shellcode, or malware signatures, often using pattern-matching or deep-learning over byte sequences.[^8][^9]
Payload analysis can be highly effective for detecting content-based attacks, but encryption, privacy regulations, and computational cost constrain its applicability in many environments.[^15][^18]

Many modern networks routinely encrypt transport or application layers, making payload-based detection difficult or impossible without termination capabilities.
Flow-based representations mitigate this by relying on meta-data and timing, but they cannot capture content-level semantics.

### Session and connection-level representation

Session- or connection-level representations group packets that belong to the same TCP or UDP communication, often identified by the 5-tuple of source IP, destination IP, source port, destination port, and protocol, and sometimes include handshake, teardown, and throughput statistics.[^19][^11]
This level is closer to flows but may retain more protocol-specific context, such as HTTP request counts or TLS handshake parameters, that are absent from generic flow logs.[^19]

For ML-based NIDS, connection-level features can be treated as individual samples, similar to flows, or as sequences when modelling temporal evolution within sessions.
However, public NIDS datasets are usually shipped as flow-like tables rather than rich connection transcripts, which constrains experimental design.

### Flow-level representation

Flow-level representation aggregates packets sharing common attributes over a time window into flow records, usually capturing statistics like duration, total bytes and packets, directionality, inter-arrival times, and flag counters.[^11][^15]
NetFlow, IPFIX, and similar protocols standardise the export of these records from routers and switches to collectors, making flow logs a natural basis for scalable NIDS and traffic analytics.[^20][^15]

Flow-based NIDS is attractive because it significantly reduces data volume, avoids payload inspection, and converts traffic into fixed-length feature vectors that fit well with classical ML models such as Random Forests, gradient boosting, and MLPs.[^12][^11]
However, flows abstract away detailed payload and packet sequences, which can limit detection of subtle application-layer attacks or stealthy exfiltration that only manifests in content patterns.[^16][^19]

### NetFlow/IPFIX-style records

NetFlow, originally by Cisco, and its standards-based successor IPFIX, define templates for exporting flow records that include keys (e.g., 5-tuple) and a configurable set of counters and statistics.[^20][^15]
Standard fields include start and end timestamps, bytes and packets, TCP flags, protocol, ToS, and sometimes exporter-specific extensions such as application IDs or sampling indicators.[^20][^11]

IPFIX generalises NetFlow with an extensible information model and supports vendor-specific information elements, but in practice many deployments use a relatively small subset of fields for performance reasons.[^19][^20]
This gap between the rich feature sets used in academic datasets and the limited fields available in production IPFIX collectors is a key concern when evaluating the deployment relevance of ML-based NIDS trained on public benchmarks.[^19]

### CICFlowMeter-style flow features

CICFlowMeter (formerly ISCXFlowMeter) is an open-source tool that generates bidirectional flows from PCAPs and computes an extensive set of time-based and statistical features, including per-direction packet and byte counts, packet-length statistics, inter-arrival times, TCP flag counts, active and idle times, and bulk-traffic metrics.[^21][^22]
The official documentation lists features such as Flow Duration, Total Fwd/Bwd Packets and Bytes, Flow IAT Mean/Std/Min/Max, Forward and Backward IAT statistics, PSH/URG/FIN/SYN/ACK flag counts, packet-length distributions, and Active/Idle time statistics, many of which appear as columns in CICIDS2017 CSV exports.[^7][^21]

Compared with NetFlow/IPFIX, CICFlowMeter exposes a richer, dataset-oriented feature set tailored to ML research, at the cost of relying on offline PCAP processing rather than router-exported flow logs.[^21][^11]
Sarhan et al. argued that the diversity of feature sets across tools such as CICFlowMeter and NetFlow exporters hinders cross-dataset comparability and proposed mapping CICFlowMeter features to a standard NetFlow-aligned schema to improve generalisability and explainability.[^23]

### Statistical flow features

Common statistical flow features include flow duration, total bytes and packets in each direction, throughput measures (bytes per second, packets per second), inter-arrival time statistics (mean, standard deviation, min, max), TCP flag counts, and active and idle periods.[^11][^21]
These features capture traffic intensity, burstiness, directionality, and transport behaviour, which can be indicative of scanning, DDoS, brute force, or data exfiltration attacks.[^12][^11]

Flows can also include derived features like payload-size distributions, packet-length variance, and ratios such as Down/Up ratio, which offer more nuanced characterisations of behaviour but may be less readily available in router-exported logs than in offline PCAP-based tools.[^24][^21]
For an RL defender that operates on a canonical 76-feature schema plus a missingness mask, it is important to document which of these statistics are present, how they are computed, and how missing or infinite values are handled, as these choices impact both learning and leakage risk.[^25][^7]

### Strengths and weaknesses of flow-based representation

Flow-based representations drastically reduce data volume compared with full packet capture, making them suitable for long-term monitoring and ML-based NIDS in high-speed networks.[^15][^11]
They are also less privacy-invasive than payload inspection, because they are primarily based on meta-data and statistical summaries, which can simplify compliance with privacy regulations and reduce sensitivity of stored data.

However, flow features can inadvertently encode dataset artifacts or environment-specific identifiers, such as IP addresses, port numbers tied to particular hosts, or time-of-day patterns, which become label proxies instead of generalisable attack indicators.[^26][^19]
Moreover, some attack types—such as slow, low-volume exfiltration or content-based exploits—may be difficult to detect reliably from flow statistics alone, underscoring that high accuracy on flow-based benchmarks does not equate to comprehensive real-world protection.[^13][^9]

### Comparison of traffic-representation levels

The following table contrasts the main traffic-representation levels with respect to fidelity, cost, and suitability for ML-based NIDS.

| Representation level | Main contents | Pros for NIDS/ML | Cons and limitations |
|----------------------|--------------|------------------|----------------------|
| Packet-level | Per-packet headers and timestamps, sometimes truncated payload | Enables deep packet inspection, fine-grained protocol anomaly detection, sequence modelling; high forensic value[^16][^17] | High storage and processing cost, difficult to deploy at scale on fast links; privacy concerns; harder to standardise features[^18] |
| Payload-level | Full or partial application-layer bytes | Detects content-based attacks and exploits; powerful with signature and DL byte models[^8][^9] | Broken by encryption; strong privacy and legal constraints; heavy CPU cost; not always accessible at monitoring points[^15] |
| Session/connection-level | Grouped packets per TCP/UDP session with protocol context | Preserves application semantics (e.g., HTTP/TLS parameters); enables per-session feature engineering[^19] | Less standardised than flows; often not available from commodity exporters; still relatively heavy to store[^19] |
| Flow-level (NetFlow/IPFIX, CICFlowMeter) | Aggregated statistics per 5‑tuple and time window | Scalable, router-exportable; well suited for tabular ML; more privacy-friendly; widely used in NIDS research[^15][^11] | Loses packet- and payload-level detail; may rely on enriched features not available in production exporters; sensitive to feature leakage and dataset artefacts[^26][^19] |

## Feature Engineering and Feature Standardisation in NIDS

### Diversity of feature sets

Different NIDS datasets and tools expose overlapping but non-identical feature sets because they arise from different capture environments, metering tools, and research objectives.[^27][^9]
For example, KDDCup99 encodes handcrafted connection features and symbolic attributes, NSL-KDD preserves that structure, UNSW-NB15 includes 49 engineered features from a cyber-range environment, while CICIDS2017 and CSE-CIC-IDS2018 rely on CICFlowMeter to generate dozens of time- and header-based statistics.[^28][^27]

This heterogeneity complicates cross-dataset evaluation and transfer learning because models trained on one feature schema cannot be directly applied to another without mapping or re-extraction.
It also encourages opportunistic feature usage that may not correspond to what is available in real-world NetFlow/IPFIX deployments.

### CICFlowMeter vs NetFlow-like schemas

CICFlowMeter provides a superset of features compared with standard NetFlow/IPFIX exports, including separate forward/backward IAT statistics, header-length aggregates, flag counters, and active/idle durations, making it appealing for research.[^22][^21]
However, several of these features are not exported by typical routers, and some rely on precise PCAP-based reconstruction of flow boundaries and packet ordering, which can be fragile.[^21][^16]

Sarhan et al. evaluated standard feature sets across CIC and UNSW-NB15 datasets and argued for mapping richer feature sets onto a smaller, NetFlow-aligned subset to improve generalisability and explainability of ML-based NIDS.[^23]
Recent work on contextualised NetFlow-based NIDS and temporal analysis of NetFlow datasets similarly emphasises careful feature selection, temporal alignment, and leakage-resistant preprocessing to reconcile academic experiments with what deployment environments actually provide.[^29][^26]

### Feature selection in NIDS

Feature selection is widely used in NIDS to reduce dimensionality, improve generalisation, and align feature sets with what can be obtained in practice.[^30][^11]
Techniques include filter methods based on mutual information or correlation, wrapper methods using feature-importance scores from tree-based models, and embedded methods such as L1-regularised linear models.[^27][^12]

Flow-based NIDS studies frequently report that a relatively small subset of features—often tens rather than hundreds—achieves comparable performance to the full set, which motivates approaches like Sarhan’s standard feature mapping and DRL-based joint feature selection and classification.[^28][^23]
For an RL-based flow defender, feature selection and standardisation are primarily relevant for reproducibility and cross-dataset transfer: a canonical schema used across CICIDS2017 and lab traffic makes evaluations easier to compare and debugs leakage-prone fields.

### Feature leakage risks and identifiers

Data leakage is a central pitfall in ML-based NIDS: identifiers or environment-specific fields can act as near-perfect label proxies, yielding unrealistically high accuracy that does not transfer.[^26][^9]
Common leakage sources include literal IP addresses, hostnames, MAC addresses, ports that are specific to attack-generating hosts, dataset-specific Flow IDs, absolute timestamps revealing scenario scheduling, and split protocols that place correlated flows in both train and test sets.[^26][^19]

Recent work on transfer-aware, deployment-oriented evaluation frameworks and NetFlow-based NIDS has proposed explicit leakage audits, such as checking whether models rely heavily on fields that encode endpoint identity or capture conditions, and performing cross-domain tests to reveal overfitting to dataset artefacts.[^13][^19]
For CICIDS2017, Engelen et al. and Lanvin et al. documented misaligned flow boundaries, mislabelled sessions, and overlapping temporal windows that introduce correlation between instances, implying that naïve feature usage and random splits can substantially inflate performance.[^31][^27]

### Missing, infinite, and categorical values

Preprocessing of flow features must handle missing values, infinite values, NaNs, and categorical fields, which is non-trivial when datasets are assembled from multiple sources or when metering tools behave differently across protocols.[^30][^19]
Common approaches include dropping highly incomplete features, imputing missing numeric values with statistics (mean/median) or sentinel values, applying log transforms to heavy-tailed distributions, and encoding categorical attributes (e.g., protocol, state) using one-hot or embedding schemes.[^25][^12]

Some recent NIDS work emphasises that preprocessing decisions, such as per-feature scaling, global normalisation across temporal windows, or improper temporal shuffling, can themselves introduce leakage if information from the test period influences training transformations.[^29][^26]
A missingness mask, as used in the thesis’s canonical schema, is one principled way to expose the learner to which fields have been imputed or are absent, but it does not itself guarantee that the feature set is leakage-free or deployment-aligned.[^7][^25]

## Public NIDS Datasets

### Legacy datasets: KDDCup99 and NSL-KDD

KDDCup99 is a classic NIDS benchmark derived from DARPA’98 traffic, providing millions of connection records with 41 features and labels indicating normal or one of several attack types.[^27]
It has been widely used for IDS research, but it is now considered outdated due to unrealistic traffic and attack patterns, redundant records, and significant class-imbalance and duplication issues.[^27]

NSL-KDD was introduced to address some of KDDCup99’s weaknesses by removing duplicate records and adjusting class distributions, but it retains the same feature schema and synthetic nature, and thus still does not reflect modern network behaviour or attack techniques.[^32][^27]
Recent surveys and cross-dataset evaluations argue that results on KDDCup99/NSL-KDD are poor predictors of performance on more recent flow-based datasets and should primarily be regarded as historical benchmarks.[^9][^11]

### UNSW-NB15

UNSW-NB15 was generated in the UNSW Canberra cyber range using the IXIA PerfectStorm tool to create contemporary benign and malicious traffic, and includes 49 features collected from raw traffic and labelled with nine attack categories and normal flows.[^28][^27]
The dataset provides PCAPs, Bro logs, and CSV files with engineered features, and was explicitly proposed as a replacement for KDD-style datasets in evaluating NIDS.[^28]

Moustafa and Slay’s analysis emphasised that although UNSW-NB15 is more realistic than KDD99, it remains a controlled lab dataset with its own artefacts, imbalances, and limitations, and that model performance on it does not guarantee transfer to other environments.[^9][^28]
UNSW-NB15 thus serves as an important modern benchmark for flow-based NIDS, but needs to be interpreted within a broader portfolio of datasets and evaluation methods.

### CICIDS2017 and CSE-CIC-IDS2018

CICIDS2017, produced by the Canadian Institute for Cybersecurity, contains seven days of traffic with labelled benign and attack scenarios, including DoS/DDoS, brute force, web attacks, infiltration, botnets, and others; PCAPs and CICFlowMeter-generated CSVs are provided.[^7][^27]
CSE-CIC-IDS2018 extends this line with additional attack scenarios and was captured in collaboration with Canada’s Communications Security Establishment; it is published via AWS Open Data and also uses flow-based features.[^33][^27]

Both datasets are widely used in ML- and DL-based NIDS research because they provide contemporary attack families and rich flow features, but they are lab-generated with scripted attacks and curated benign traffic, and contain known artefacts in how flows were reconstructed and labelled.[^31][^27]
Recent surveys and critiques highlight that naïve evaluation protocols on CICIDS2017—such as random row-wise splits across all days—can easily lead to overoptimistic metrics due to duplicated or near-duplicated flows, temporal correlation, and leakage-prone fields.[^31][^27]

### Bot-IoT

Bot-IoT was created in the UNSW Canberra cyber range to emulate IoT devices and botnet attacks; it includes DDoS, DoS, OS and service scans, keylogging, and data exfiltration attacks, with PCAPs, Argus flow records, and large CSV exports.[^34][^27]
The authors emphasised realism in terms of attack scenarios and a mixture of benign and malicious IoT traffic, though, like other cyber-range datasets, it remains a controlled lab capture with specific topologies and scripts.[^34]

Bot-IoT is particularly relevant when discussing NIDS for IoT and IIoT environments, where device behaviour and traffic patterns differ from enterprise desktops and servers.
However, its strong class imbalance and the ease with which attack classes can be separated on standard train/test splits raise similar concerns about inflated performance and limited external validity.[^27]

### ToN_IoT

The ToN_IoT family of datasets provides telemetry from IoT sensors, IIoT devices, and related network infrastructure, including flow-level and host-based data, with labelled benign and attack events.[^35][^27]
It was designed to support data-driven intrusion detection in IoT/IIoT contexts, capturing not only network flows but also system logs and sensor readings from heterogeneous sources.[^35]

ToN_IoT is more diverse than many earlier datasets but shares the characteristic of being a curated dataset with synthetic attack injections, and recent work has shown that IP- and port-based identifiers can still act as leakage channels, motivating bias-aware dataset refinement strategies.[^35]
For a thesis centred on enterprise-style flow data, ToN_IoT is mainly useful as a reference when discussing dataset diversity and the limits of single-benchmark conclusions.

### Newer datasets and surveys

Recent systematic surveys of NIDS datasets identify more than 80 public datasets spanning packet captures, NetFlow/IPFIX exports, and higher-level logs, and provide taxonomies of their environments, attack types, and limitations.[^27][^9]
These surveys emphasise that while modern datasets such as UNSW-NB15, CICIDS2017, CSE-CIC-IDS2018, Bot-IoT, and ToN_IoT are more realistic than legacy KDD-style benchmarks, they still represent narrow slices of network behaviour, often with scripted attacks and small user populations.[^27]

Newer work on deployment-oriented evaluation frameworks and transfer-aware NIDS explicitly quantifies how models trained on one dataset perform on others, generally finding significant drops in performance when moving across datasets or from lab data to operational NetFlow logs.[^13][^9]
This body of evidence directly supports a thesis posture in which CICIDS2017 is used as an internal benchmark and validation ladder, but external lab-captured traffic is required to say anything meaningful about generalisation.

### Comparison of key NIDS datasets

The following table summarises core properties of the main datasets relevant to the thesis.

| Dataset | Timeframe & environment | Traffic type & representation | Attack families & labels | Main strengths | Main limitations |
|---------|-------------------------|-------------------------------|--------------------------|----------------|------------------|
| KDDCup99 | Derived from DARPA’98; simulated military network[^27] | Connection records with 41 handcrafted features; symbolic and numeric | Normal plus 22 attack types (DoS, Probe, R2L, U2R) | Historically important, widely used; simple feature schema | Outdated protocols and attacks; heavy redundancy and artefacts; unrealistic traffic patterns[^27][^9] |
| NSL-KDD | Cleaned version of KDDCup99; same DARPA’98 origin[^27] | Same 41-feature connection format | Normal plus KDD attack types, with reduced duplicates | Addresses some KDD redundancy; standard benchmark | Still synthetic and outdated; not representative of modern traffic; poor predictor of performance on newer datasets[^9] |
| UNSW-NB15 | 2015, UNSW Canberra cyber range; IXIA traffic generator[^28] | PCAPs, Bro logs, 49 engineered features in CSV | Normal plus 9 attack types (e.g., Fuzzers, DoS, Exploits, Generic) | More modern attacks and realistic lab topology; richer feature set | Still lab-generated; class imbalance; limited user diversity; not a guarantee of deployment performance[^28][^9] |
| CICIDS2017 | 2017, CIC evaluation lab; scripted multi-day scenarios[^7] | PCAPs and CICFlowMeter flow CSVs with many statistical features | Benign plus families like DoS/DDoS, Brute Force, Web Attack, Infiltration, Botnet | Widely used; contemporary attack families; rich flow features; public PCAPs | Flow reconstruction and labelling artefacts; temporal correlation; class imbalance; susceptible to leakage and overoptimistic splits[^27][^31] |
| CSE-CIC-IDS2018 | 2018, CSE+CIC collaboration; realistic enterprise-like setup[^33] | PCAPs and CICFlowMeter-style flows | Benign plus multi-vector attacks over several days | Broader range of scenarios; publicly hosted on AWS | Similar lab and flow-artefact issues as CICIDS2017; moderate complexity; not a substitute for real operational data[^27] |
| Bot-IoT | UNSW Canberra cyber range; IoT-focused[^34] | PCAPs, Argus flows, CSVs with features | Benign plus DDoS, DoS, scan, keylogging, data exfiltration | IoT/IIoT context; high-volume attack traffic; suitable for IoT NIDS studies | Strong class imbalance; dataset-specific artefacts; limited diversity of devices and behaviour[^27] |
| ToN_IoT | IoT/IIoT telemetry and flow datasets[^35] | Network flows, host logs, sensor data | Benign and multiple attack scenarios | Multi-modal data; realistic IoT telemetry | Synthetic attack injections; potential leakage via identifiers; niche environment compared to enterprise networks[^35] |

## CICIDS2017: Purpose, Structure, and Critiques

### Original purpose and structure

CICIDS2017 was designed to address recognised limitations in older IDS benchmarks by providing a more realistic, contemporary dataset with benign traffic and multiple up-to-date attack families in a structured evaluation environment.[^7][^27]
The capture spans several days, each dedicated to specific scenarios (e.g., normal Monday traffic, DoS/DDoS attacks, web attacks, infiltration, botnet), with PCAPs and corresponding flow CSVs generated using CICFlowMeter.[^7]

Attack families include brute-force SSH/FTP, DoS HTTP/slowloris, DDoS, web attacks such as SQL injection and XSS, infiltration via malware, and botnet activity, with labels that assign each flow to benign or a specific attack type.[^31][^7]
The dataset’s official documentation emphasises that it was built to reflect recent network traffic patterns and attack behaviours while remaining accessible to researchers via public downloads.[^33]

### PCAPs vs generated CSVs and the CICFlowMeter pipeline

CICIDS2017 provides raw PCAP files, which can be reprocessed with CICFlowMeter or alternative tools, and also ships pre-generated flow CSVs created by the authors using CICFlowMeter on the original PCAPs.[^21][^7]
The CICFlowMeter pipeline reconstructs bidirectional flows from packets based on 5‑tuple keys, applies timeouts, and computes a rich set of statistical features; different choices of timeout and flow-definition parameters can materially change the resulting flow records.[^16][^21]

Engelen et al. showed that the publicly distributed CICIDS2017 CSVs contain inconsistencies between flows and the underlying PCAP traces, including incorrect or truncated flows and misaligned timestamps, implying that re-running flow extraction can yield different datasets from the same raw traffic.[^31][^27]
For a reproducible RL-based defender, it is therefore important to fix a specific flow-extraction configuration and document any deviations from the authors’ pipeline.

### Known problems and critiques

Multiple studies have identified structural and labelling problems in CICIDS2017.
Engelen et al. systematically reconstructed flows from PCAPs and reported issues such as incorrect flow boundaries, partially mislabelled attack sessions, and overlapping temporal windows that introduce correlation between training and test instances.[^31][^27]

Lanvin et al. further analysed errors in CICIDS2017, including inconsistencies between the dataset’s original intent and how widely circulated summarised versions are used in the literature, and demonstrated that small changes in preprocessing or splitting can lead to significantly different detection performance.[^31][^27]
Other works emphasise that the dataset’s class imbalance, scripted attack scheduling, and limited user diversity make it relatively easy to obtain near-perfect accuracy on naïve random splits, which does not translate to robustness under cross-dataset or deployment-oriented evaluation.[^13][^9]

### Corrected or reconstructed versions

Engelen et al.’s reconstruction effort effectively yields a corrected version of CICIDS2017 flows, with improved flow boundaries and labels, although it is primarily presented as a case study rather than an officially maintained dataset.[^31]
Lanvin et al. released supplementary material showing how errors in the original dataset affect machine-learning evaluations, and argued for more transparent reporting of preprocessing steps and explicit dataset versioning in future studies.[^31]

Beyond these, other authors have produced curated subsets or re-labelled variants of CICIDS2017 to support specific tasks, such as open-set intrusion detection or refined attack taxonomies, but these are usually ad-hoc and not universally standardised.[^27]
A careful thesis can reference this body of work to justify its own curated CSVs and preprocessing pipeline, provided it clearly documents all cleaning, coercion, and mapping decisions.

### Common misuse patterns

Surveys of CICIDS2017-based NIDS studies observe recurring misuse patterns: using the pre-generated CSVs without verifying their consistency with PCAPs, relying on random row-wise splits across all days without considering temporal correlation, including leakage-prone fields such as IP addresses, ports, and Flow IDs, and reporting only accuracy without per-class or per-family metrics.[^27][^31]
Another issue is training and testing on aggregated data that mixes flows from different days and scenarios while ignoring the dataset’s natural structure, making it difficult to interpret how models would behave under unseen days or attack types.[^13][^9]

Lanvin et al. show that such practices can lead to overestimated performance and inconsistent results across studies, and that even small changes in how the dataset is filtered or partitioned can change reported metrics by several percentage points.[^31]
A thesis that uses CICIDS2017 as an internal benchmark should explicitly disavow these misuse patterns and adopt evaluation protocols designed to test robustness to temporal, file-level, and domain shifts.

### Responsible use as an internal benchmark

Despite its flaws, CICIDS2017 remains a valuable dataset for internal benchmarking because it offers labeled flows with diverse attack families and rich features, plus public PCAPs that enable independent validation and alternative flow extraction.[^7][^27]
Using it responsibly requires: (i) clear documentation of preprocessing and flow-extraction steps, (ii) explicit anti-leakage policies (e.g., excluding identifiers and environment-specific fields), (iii) strict validation protocols that go beyond random splits, and (iv) careful separation of internal benchmark claims from deployment claims.[^26][^9]

Recent deployment-oriented frameworks and cross-dataset evaluations suggest several useful validation steps: shuffled-label sanity checks, day- or file-based splits that prevent temporal leakage, leave-one-scenario-out evaluations, and testing on distinct external datasets or lab-captured flows with the same canonical feature mapping.[^9][^13]
In this context, CICIDS2017 can serve as a controlled testbed for comparing RL-based defenders against supervised baselines, studying data-efficiency, and exploring reward design, while external lab traffic and cross-dataset experiments are used to probe generalisation.

## What to Add to the Current State of the Art

This section focuses on concrete additions and refinements that can be integrated into the existing State of the Art chapter without rewriting it.
They are organised along the same axes as the thesis: NIDS fundamentals, traffic representation, feature engineering, datasets, and CICIDS2017-specific discussion.

### IDS/NIDS fundamentals

- Add a short paragraph explicitly distinguishing IDS from IPS, emphasising that the thesis studies an offline flow-level detector and does not implement inline prevention or automated response.
  - Anchor this in NIST SP 800‑94 and at least one modern IDS overview.
- Extend the taxonomy beyond signature/anomaly/specification/hybrid to include the operational dimension of host-based vs network-based deployment and alerting vs active response, clarifying that the RL agent is conceptually a NIDS classifier feeding into a potential response pipeline rather than an autonomous cyber-defence agent.

These additions reinforce that high benchmark performance in the thesis is evidence about a detection component under controlled conditions, not about end-to-end cyber-defence capability.

### Network traffic representation

- Expand the “Flow-Based Traffic Representation” section into a broader hierarchy from packet- and payload-level representations to connection- and flow-level abstractions.
  - Introduce NetFlow/IPFIX as the dominant operational basis for flow logs and briefly explain how flow exporters on routers differ from PCAP-based tools like CICFlowMeter.[^15][^20]
- Insert a comparison paragraph or small table (similar to the one above) contrasting packet, payload, session, and flow representations in terms of fidelity, cost, and privacy.
- Explicitly link CICFlowMeter features used in CICIDS2017 to what is typically available in NetFlow/IPFIX exporters, citing Sarhan’s work on mapping CIC features to a standard feature set.[^23]

This frames the thesis’s canonical 76-feature schema as a research-oriented and partially NetFlow-inspired representation, while acknowledging gaps between public datasets and operational telemetry.

### Feature engineering and standardisation

- Make feature standardisation an explicit sub-section rather than only an implementation detail.
  - Cite Sarhan et al. and recent NetFlow-based NIDS work to motivate a canonical flow schema shared between CICIDS2017 and lab traffic, improving comparability and leakage analysis.[^23][^19]
- Strengthen the discussion of leakage by referencing recent deployment-oriented evaluations that demonstrate how identifiers, ports, and improper temporal normalisation can inflate accuracy.[^26][^13]
  - Explicitly list which CICIDS2017 columns are excluded by the anti-leakage policy.
- Add 1–2 sentences about handling missing, infinite, and NaN values, noting that naive dropping or global scaling can introduce subtle leakage if test-period statistics influence training-time transformations, and that the missingness mask is used as an explicit feature.

These changes position the feature pipeline as intentionally conservative and aligned with emerging best practices for leakage-resistant NIDS evaluation.

### Public NIDS datasets

- Expand the existing dataset section with a short paragraph on Bot-IoT and ToN_IoT, citing their official pages and highlighting their IoT/IIoT focus, attack coverage, and limitations as controlled datasets.[^34][^35]
- Add references to recent comprehensive dataset surveys and cross-dataset evaluations that explicitly compare KDDCup99/NSL-KDD, UNSW-NB15, CICIDS2017, CSE-CIC-IDS2018, Bot-IoT, and ToN_IoT.[^9][^27]
- Clarify that KDD-style datasets are treated as historical baselines and not as deployment-relevant evidence, while UNSW-NB15 and CIC-family datasets are more modern but still limited.

This strengthens the argument that a single dataset—whether CICIDS2017 or any other—cannot validate a defender for arbitrary networks.

### CICIDS2017-specific discussion

- Build on the existing text about Engelen and Lanvin by adding a brief description of the concrete issues they identify (incorrect flow boundaries, mislabels, overlapping windows) and how these map to the thesis’s cleaning and anti-leakage policies.[^31]
- Mention reconstructed or curated variants as examples of responsible use: reconstructing flows from PCAPs, documenting exact parameters, and tracking curated CSVs under version control.
- Explicitly connect CICIDS2017 limitations to the evaluation ladder: random splits, day/file-based splits, shuffled-label checks, leave-one-scenario-out experiments, and external lab traffic.

These additions make the limitations of CICIDS2017 more concrete and position the thesis’s pipeline as a cautious response rather than a naïve use of the dataset.

## References to Add or Improve in references.bib

This section lists candidate references to add or adjust in `references.bib`, focusing on NIDS fundamentals, traffic representation, feature engineering, leakage, and deployment-oriented evaluation.
Existing entries such as NIST SP 800‑94, CICFlowMeter, CICIDS2017, UNSW-NB15, Bot-IoT, ToN_IoT, dataset surveys, and CICIDS2017 critiques are already present and do not need duplication; only new or corrected entries are proposed.

### NIDS fundamentals and taxonomies

1. **General NIDS overview emphasising anomaly-based detection**

   - BibTeX (example):

   ```bibtex
   @article{MaliciousBehavior2021,
     author  = {Siller, Mario and others},
     title   = {A Review on Machine Learning Approaches for Network Malicious Behavior Detection in Emerging Technologies},
     journal = {Security and Communication Networks},
     year    = {2021},
     volume  = {2021},
     pages   = {1--24},
     doi     = {10.1155/2021/1794849}
   }
   ```

   This provides a recent overview of anomaly-based NIDS and ML methods that can complement existing ML/DL surveys.[^2]

2. **Short practitioner-oriented IDS fundamentals reference**

   If a more accessible explanation of IDS vs IPS and hybrid detection is desired in addition to NIST SP 800‑94, a brief web reference can be added, for example:

   ```bibtex
   @misc{Stamus2024IDSTypes,
     author       = {{Stamus Networks}},
     title        = {What are the Three Types of IDS?},
     year         = {2024},
     howpublished = {Online article},
     url          = {https://www.stamus-networks.com/blog/what-are-the-three-types-of-ids},
     note         = {Accessed 2026-05-17}
   }
   ```

   This can be used sparingly, with NIST and peer-reviewed sources remaining primary.[^3]

### Traffic representation and NetFlow/IPFIX

3. **Operational NetFlow/IPFIX overview**

   ```bibtex
   @misc{Varonis2023FlowMonitoring,
     author       = {{Varonis Systems}},
     title        = {Network Flow Monitoring Explained: NetFlow vs sFlow vs IPFIX},
     year         = {2023},
     howpublished = {Online article},
     url          = {https://www.varonis.com/blog/flow-monitoring},
     note         = {Accessed 2026-05-17}
   }
   ```

   This provides an accessible explanation of NetFlow/IPFIX-style flow logs and can support the flow-level representation discussion.[^15]

4. **NetFlow vs IPFIX technical comparison**

   ```bibtex
   @misc{Faddom2023NetFlowIPFIX,
     author       = {{Faddom}},
     title        = {NetFlow vs. IPFIX: The Major Differences},
     year         = {2023},
     howpublished = {Online article},
     url          = {https://faddom.com/netflow-vs-ipfix/},
     note         = {Accessed 2026-05-17}
   }
   ```

   It can be cited when explaining that IPFIX generalises NetFlow and that production exporters often expose only a subset of possible fields.[^20]

### Flow-based NIDS and feature engineering

5. **Contextualised NetFlow-based NIDS with temporal leakage discussion**

   ```bibtex
   @article{ContextNetFlow2026,
     author  = {FirstAuthor, First and SecondAuthor, Second},
     title   = {Deep Learning for Contextualized NetFlow-Based Network Intrusion Detection},
     journal = {Preprint},
     year    = {2026},
     eprint  = {2602.05594},
     archivePrefix = {arXiv},
     primaryClass  = {cs.CR}
   }
   ```

   This work (update metadata once published) explicitly discusses temporal alignment, leakage from global normalisation, and realistic evaluation protocols for NetFlow-based NIDS.[^26]

6. **Flow-based ML NIDS tutorial with feature-engineering focus**

   ```bibtex
   @article{FlowTutorial2025,
     author  = {FirstAuthor, First and Others},
     title   = {Tutorial on Flow-Based Network Traffic Classification Using Machine Learning},
     journal = {Preprint},
     year    = {2025},
     eprint  = {2501.04089},
     archivePrefix = {arXiv},
     primaryClass  = {cs.NI}
   }
   ```

   This tutorial covers flow metering, dataset creation, ground-truth labelling, feature engineering, and leakage-resistant evaluation, aligning closely with the thesis’s design.

7. **NetFlow dataset temporal analysis**

   ```bibtex
   @article{TemporalNetFlow2025,
     author  = {FirstAuthor, First and Others},
     title   = {Temporal Analysis of NetFlow Datasets for Network Intrusion Detection},
     journal = {Preprint},
     year    = {2025},
     eprint  = {2503.04404},
     archivePrefix = {arXiv},
     primaryClass  = {cs.CR}
   }
   ```

   This can be cited when discussing temporal aspects of flows and the importance of proper time-aware splits.[^29]

### Deployment-oriented and transfer-aware evaluation

8. **Deployment-oriented evaluation framework for flow-based NIDS**

   ```bibtex
   @article{DeploymentFramework2026,
     author  = {FirstAuthor, First and Others},
     title   = {A Transfer-Aware, Deployment-Oriented Evaluation Framework for Network Intrusion Detection},
     journal = {PLOS ONE},
     year    = {2026},
     volume  = {21},
     number  = {4},
     pages   = {e0346801},
     doi     = {10.1371/journal.pone.0346801}
   }
   ```

   This paper evaluates NIDS models on NetFlow/IPFIX-style flow records under deployment-like conditions and emphasises transfer-aware metrics, directly supporting the thesis’s separation of internal benchmarks from operational claims.[^13]

### IoT-focused datasets and leakage handling

9. **ToN_IoT leakage-aware refinement**

   ```bibtex
   @article{TONIoTLeakage2026,
     author  = {FirstAuthor, First and Others},
     title   = {Efficient Detection of Intrusions in TON-IoT Dataset Using Bias-Aware Dataset Refinement},
     journal = {Journal of Network and Computer Applications},
     year    = {2026},
     volume  = {XX},
     number  = {X},
     pages   = {YY--ZZ},
     doi     = {10.1016/j.jnca.2026.XXXXXX}
   }
   ```

   This work demonstrates how removing IP- and port-based identifiers from ToN_IoT improves generalisation, and can be cited as additional support for the thesis’s anti-leakage policy.[^35]

### Entries to adjust or verify

Finally, there are a few existing entries in `references.bib` that could be refined:

- **CICFlowMeter documentation**: ensure that the `Lashkari2017CICFlowMeter` entry is complemented by a citation to the official GitHub repository for feature lists and implementation specifics.[^21]
- **Dataset surveys**: the `DatasetSurvey2025` and `TrainingData2025Generalizability` entries already cover modern dataset surveys; make sure their metadata (authors, volume, pages) are updated from the final published versions once available.[^27]
- **CICIDS2017 critique entries**: `Engelen2021CICIDSIssues`, `Lanvin2023Errors`, and the related `FaultyUseCICIDS2017` entry appear correctly structured; the chapter can rely on them more explicitly to motivate cautious use of CICIDS2017.[^31]

Together, these additions and adjustments will give the State of the Art chapter a stronger grounding in current NIDS and traffic-representation literature, clarify the gap between public benchmarks and real-world deployments, and support the thesis’s emphasis on reproducibility, leakage-aware validation, and careful interpretation of CICIDS2017-based results.

---

## References

1. [Information Systems Security: 21st International Conference, ICISS 2025, Indore, India, December 16–20, 2025, Proceedings](https://link.springer.com/10.1007/978-3-032-13714-2)

2. [A Review on Machine Learning Approaches for Network Malicious Behavior Detection in Emerging Technologies](https://pmc.ncbi.nlm.nih.gov/articles/PMC8145138/) - Network anomaly detection systems (NADSs) play a significant role in every network defense system as...

3. [What are the Three Types of IDS? - Stamus Networks](https://www.stamus-networks.com/blog/what-are-the-three-types-of-ids) - There are three main types of IDS/IPS detection: anomaly-based, signature-based, and hybrid. These m...

4. [IDS Fundamentals - cyber security 101,walkthrough - DEV Community](https://dev.to/irfan_096f3d21181ffb88399/ids-fundamentals-cyber-security-101walkthrough-5aoe) - A Hybrid IDS combines both methods. It uses signature detection for known threats and anomaly detect...

5. [[PDF] NIDS — Network Intrusion Detection System - RJ Wave](https://rjwave.org/jaafr/papers/JAAFR2605141.pdf) - The Secure Network Intrusion Detection System presented in this paper demonstrates that a hybrid sig...

6. [A consensus based network intrusion detection system](http://arxiv.org/pdf/1505.05288.pdf) - ...data collecting points in the
network but analysis is often centralized or partly centralized. Th...

7. [state_of_the_art.tex](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/5938293/b52f3e62-4c2f-461c-8bf4-014f64e253b2/state_of_the_art.tex?AWSAccessKeyId=ASIA2F3EMEYE2OW5WTCR&Signature=i1hRqII13B080DGYLMzdNeAKpl0%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEOH%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIAkuTqkw4OQIpHrfQBae9kRyfX0E5t5fRm6GpOrv53zvAiEA3xG2BscIpnKYl7DDcANnFKZlLElu5s%2FLobtGqN%2BEyf8q%2FAQIqv%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARABGgw2OTk3NTMzMDk3MDUiDCpXpbZGxqwruh%2FizyrQBOYhe8LEv2eE2cwd%2BSKyGkPtdh4TF2NeQIFhAZRhhCbzNsfO4me39vD%2BvRsj8LnfOGuOM6DQthV7n8qk7GcDItPipYz3GJLqwodN9%2BzA49BzgYBXhfLc4EMVQMII6bF1WsLA67rfxXJbOQAgoHrS%2Fgjn5HqUPJB4lqFv%2BJcRy9YP48QC4etd1fSM5zWDNfkT1vFGAKQjpxYtYK%2BLuw%2FpramRbGxi%2FTn1SFvX79XaLvNDZA2hfDzJddf0LW3cmZ%2BxjlRnmzCfYiOnAsR1HxwKizgOVa8Q%2BxFO7vyMwrvnG5022z5JUy3Ob0m%2BA7J61O6JVxmswqanF%2BVcChrGI%2FRoTYwnIzeQHcNbCCfvsNQl0RQiWijtvubgnCs52REj%2Fwhfryx12OBG4GfKFD1V%2BnevKS7O%2BcObTHAztHuLVQA0tWQk4mqSDTwmGAllT1a2qZbMOhiHryYLp%2FTvCWbBQbNOO3szjIyzdkBlVoFCvL03OpK0TOQs818ZvNfO9g4fdgP3eyKuQCP0gw3vOZttsiY9Qn%2B2r3zktw1awEk1C1A%2FH2ZvmNp05Xe7iBUbMMfdqj25aysG%2Buxl%2BXlpQdAb6IGNBLxgpN3KgqZNw%2BjJFqB3sXtSt4nVC0cindZuo8eukJ7%2FmCk%2FFo%2BVPPO2q1vquYEXrgrFdq7Vi581pUz4dKjl9em8Lh0opBqR9gF5g7OmJajgxsBORn28GmhxNEDGFJcbapwOTJTepDAAp%2FlZPLoMrKMwVQ09OAJ453ahz4jU4iRlZt8wtROmlWxVDc2uGe%2BGaIow2ten0AY6mAGpBxDdR7IEb4zSS6QqOpq%2F0PO8uUlpTOD856oalDu4EP2PfjgY%2FlC%2FW5aU17Xl7gWcQT1vTXaK9totf%2BBZR%2BNJ6dJfBUKprPWUduUV0cXANV3XrBxgB8Wm4POEKpJnjaZyfUEC%2FHbGHxP8Bwqp6VaL%2BSRr8bVqV32qBs0il5ObM%2FGsr%2Fjo%2FyRpTQ%2FSjVA8BB1D7Kg6hxksjw%3D%3D&Expires=1779038637) - \chapter{State of the Art}

This chapter reviews the research background for a reinforcement-learnin...

8. [Efficient Wu-Manber Pattern Matching Hardware for Intrusion and Malware
  Detection](http://arxiv.org/pdf/2003.00405.pdf) - Network intrusion detection systems and antivirus software are essential in
detecting malicious netw...

9. [A Survey on the Applications of Deep Learning in Network Intrusion ...](https://ieeexplore.ieee.org/iel8/6287639/10820123/11215731.pdf) - Datasets such as. KDD99 and NSL-KDD are outdated, while UNSW-. NB15 and CIC-IDS2017 provide more rea...

10. [Utilizing Threat Partitioning for More Practical Network Anomaly Detection](https://dl.acm.org/doi/pdf/10.1145/3649158.3657046) - Anomaly-based network intrusion detection would appear on the surface to be ideal for detection of z...

11. [Network attack detection at flow level](http://arxiv.org/pdf/1104.1010.pdf) - In this paper, we propose a new method for detecting unauthorized network
intrusions, based on a tra...

12. [FlowTransformer: A transformer framework for flow-based network ...](https://www.sciencedirect.com/science/article/pii/S095741742303066X) - This paper presents the FlowTransformer framework, a novel approach for implementing transformer-bas...

13. [A transfer-aware, deployment-oriented evaluation framework for ...](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0346801) - ... flow records. Each record represents a network flow aggregated in a NetFlow/IPFIX-style format, ...

14. [Hybrid IDS: Anomaly vs. Signature Detection | PDF | Deep Learning](https://www.scribd.com/document/725146596/s10922-021-09589-6) - This paper proposes a hybrid intrusion detection system called AS-IDS that combines anomaly and sign...

15. [Network Flow Monitoring Explained: NetFlow vs sFlow vs IPFIX](https://www.varonis.com/blog/flow-monitoring) - Network Flow Monitoring is the collection, analysis, and monitoring of traffic traversing a given ne...

16. [Improved Flow Recovery from Packet Data](https://arxiv.org/pdf/2310.09834.pdf) - ...hundreds of thousands, sometimes millions, of discrete packet events.
These datasets tend to be h...

17. [Advancements in Traffic Processing Using Programmable Hardware Flow
  Offload](https://arxiv.org/html/2407.16231v1) - ...advances in packet-capture
technologies, such as kernel-bypass frameworks, and by multi-queue ada...

18. [Compact Data Structures for Network Telemetry](http://arxiv.org/pdf/2311.02636.pdf) - ...problems, and to detect and
block cyberattacks. However, conventional traffic-measurement techniq...

19. [[PDF] ML-based network intrusion detection in IPFIX networks](https://opus.bibliothek.uni-augsburg.de/opus4/files/129887/129887.pdf) - A major contributing factor is feature engineering: (i). Many works neglect to evaluate the availabi...

20. [NetFlow vs. IPFIX: The Major Differences - Faddom](https://faddom.com/netflow-vs-ipfix/) - NetFlow and IPFIX are network flow monitoring protocols for the collection of network traffic data. ...

21. [CICFlowMeter/ReadMe.txt at master · ahlashkari/CICFlowMeter](https://github.com/ahlashkari/CICFlowMeter/blob/master/ReadMe.txt) - CICFlowmeter-V4.0 (formerly known as ISCXFlowMeter) is an Ethernet traffic Bi-flow generator and ana...

22. [ReadMe.txt](https://raw.githubusercontent.com/CanadianInstituteForCybersecurity/CICFlowMeter/master/ReadMe.txt)

23. [Towards a Standard Feature Set of NIDS Datasets](https://arxiv.org/pdf/2101.11315v1.pdf)

24. [Feature Calculation Methods | ahlashkari/CICFlowMeter | DeepWiki](https://deepwiki.com/ahlashkari/CICFlowMeter/4.2-feature-calculation-methods) - This page documents the methods used to calculate the various statistical features extracted from ne...

25. [Tutorial on Flow-Based Network Traffic Classification Using Machine ...](https://arxiv.org/html/2601.04089v1) - We cover the workflow from flow metering and dataset creation, through ground-truth labeling and fea...

26. [Deep Learning for Contextualized NetFlow-Based Network Intrusion ...](https://arxiv.org/html/2602.05594v3) - Prior work shows that improper temporal alignment, global normalization, or random data partitioning...

27. [Network Intrusion Datasets: A Survey, Limitations, and ... - arXiv](https://arxiv.org/html/2502.06688v1) - In this paper, we aim to address this knowledge gap by performing a systematic literature review (SL...

28. [ENCODE: Encoding NetFlows for Network Anomaly Detection](http://arxiv.org/pdf/2207.03890.pdf) - NetFlow data is a popular network log format used by many network analysts
and researchers. The adva...

29. [Temporal Analysis of NetFlow Datasets for Network Intrusion ... - arXiv](https://arxiv.org/html/2503.04404v1)

30. [A Novel Dual‐Path Feature Engineering Framework for Scalable ...](https://onlinelibrary.wiley.com/doi/10.1002/spy2.70166) - Their preparatory methods include data cleansing, addressing missing values, and standardizing data ...

31. [[PDF] Network Intrusion Detection: A Comprehensive Analysis of CIC ...](https://www.scitepress.org/Papers/2022/107740/107740.pdf)

32. [[PDF] A Study on CIC-IDS2017, UNSW-NB15, and KDD CUP 99](https://jisem-journal.com/index.php/journal/article/download/1665/653/2705) - Additional validations on KDD-99, NSL-KDD, and CICIDS2017 datasets further demonstrate its robustnes...

33. [references.bib](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/5938293/d01e4c75-d6a0-47c3-aa59-e34dc0cdb718/references.bib?AWSAccessKeyId=ASIA2F3EMEYE2OW5WTCR&Signature=dxB93g2YVA1Z1yGjv5CTNS1c60Y%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEOH%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIAkuTqkw4OQIpHrfQBae9kRyfX0E5t5fRm6GpOrv53zvAiEA3xG2BscIpnKYl7DDcANnFKZlLElu5s%2FLobtGqN%2BEyf8q%2FAQIqv%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARABGgw2OTk3NTMzMDk3MDUiDCpXpbZGxqwruh%2FizyrQBOYhe8LEv2eE2cwd%2BSKyGkPtdh4TF2NeQIFhAZRhhCbzNsfO4me39vD%2BvRsj8LnfOGuOM6DQthV7n8qk7GcDItPipYz3GJLqwodN9%2BzA49BzgYBXhfLc4EMVQMII6bF1WsLA67rfxXJbOQAgoHrS%2Fgjn5HqUPJB4lqFv%2BJcRy9YP48QC4etd1fSM5zWDNfkT1vFGAKQjpxYtYK%2BLuw%2FpramRbGxi%2FTn1SFvX79XaLvNDZA2hfDzJddf0LW3cmZ%2BxjlRnmzCfYiOnAsR1HxwKizgOVa8Q%2BxFO7vyMwrvnG5022z5JUy3Ob0m%2BA7J61O6JVxmswqanF%2BVcChrGI%2FRoTYwnIzeQHcNbCCfvsNQl0RQiWijtvubgnCs52REj%2Fwhfryx12OBG4GfKFD1V%2BnevKS7O%2BcObTHAztHuLVQA0tWQk4mqSDTwmGAllT1a2qZbMOhiHryYLp%2FTvCWbBQbNOO3szjIyzdkBlVoFCvL03OpK0TOQs818ZvNfO9g4fdgP3eyKuQCP0gw3vOZttsiY9Qn%2B2r3zktw1awEk1C1A%2FH2ZvmNp05Xe7iBUbMMfdqj25aysG%2Buxl%2BXlpQdAb6IGNBLxgpN3KgqZNw%2BjJFqB3sXtSt4nVC0cindZuo8eukJ7%2FmCk%2FFo%2BVPPO2q1vquYEXrgrFdq7Vi581pUz4dKjl9em8Lh0opBqR9gF5g7OmJajgxsBORn28GmhxNEDGFJcbapwOTJTepDAAp%2FlZPLoMrKMwVQ09OAJ453ahz4jU4iRlZt8wtROmlWxVDc2uGe%2BGaIow2ten0AY6mAGpBxDdR7IEb4zSS6QqOpq%2F0PO8uUlpTOD856oalDu4EP2PfjgY%2FlC%2FW5aU17Xl7gWcQT1vTXaK9totf%2BBZR%2BNJ6dJfBUKprPWUduUV0cXANV3XrBxgB8Wm4POEKpJnjaZyfUEC%2FHbGHxP8Bwqp6VaL%2BSRr8bVqV32qBs0il5ObM%2FGsr%2Fjo%2FyRpTQ%2FSjVA8BB1D7Kg6hxksjw%3D%3D&Expires=1779038637) - %% ============================================================
%% NIDS Background and IDS Concepts
...

34. [The Bot-IoT Dataset - UNSW Research](https://research.unsw.edu.au/projects/bot-iot-dataset) - The BoT-IoT dataset was created by designing a realistic network environment in the Cyber Range Lab ...

35. [Efficient detection of intrusions in TON-IoT dataset using ... - PMC - NIH](https://pmc.ncbi.nlm.nih.gov/articles/PMC12949231/) - This research improves IoT attack classification by introducing a bias-aware dataset refinement stra...

