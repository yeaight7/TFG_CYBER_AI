# State-of-the-Art Supplementary Draft
## Second revised pass — 2026-05-16

This file provides additional draft content for the thesis chapter "State of the Art," organized
as flowing academic prose for direct integration into `report/drafts/state_of_the_art.md` after
human review. Each section is labelled with the existing draft section it supplements.

**Citation discipline.** Keys confirmed in `report/references.bib` are used without annotation.
Keys from `Research/Research2.md` candidate bibliography requiring DOI verification before bib
addition are marked `[CANDIDATE]`. Keys not yet in any bibliography and requiring external
verification are marked `[VERIFY]`. No DOIs, author names, or years are invented.

**What the existing draft already covers.** The draft at `report/drafts/state_of_the_art.md`
is a complete 12-section chapter. Sections 1–7 and 10–12 require only minor revision. The
additions below address specific named gaps: CICIDS2017 quality concerns (Section 4), canonical
schema motivation (Section 5), named RL-IDS prior works (Section 8), dataset-as-environment
precedent (Section 9), data efficiency (new Section 13), ACD boundary (Section 8 or 12), and
foundational evaluation limits (Section 10).

---

## Section A — CICIDS2017 Quality Concerns
*Supplements draft Section 4; insert after the paragraph beginning "The limitations of CICIDS2017 are central to the thesis argument."*

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

---

## Section B — Canonical Feature Schema and the Case for a Fixed Representation
*Supplements draft Section 5; insert after the paragraph on Random Forest as a baseline.*

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

---

## Section C — Reinforcement Learning for Intrusion Detection: Named Prior Works
*Supplements draft Section 8; replaces the paragraph with [VERIFY] placeholders for specific prior works.*

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

---

## Section D — The Dataset-as-Environment Design: Named Precedent
*Supplements draft Section 9; insert after the opening paragraph on the central methodological tension.*

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

---

## Section E — Data Efficiency and Training-Scale Evaluation
*New section; add as Section 13 or absorb into the positioning section.*

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

---

## Section F — Foundational Evaluation Limits
*Supplements draft Section 10; insert before the paragraph on random row-wise splits.*

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

---

## Citation Notes

| Section | Key | Status | Note |
|---|---|---|---|
| A | `Engelen2021CICIDSIssues` | VERIFY | Authors likely Engelen, Rimmer, Latré et al.; venue likely 2021 security/networking conference; DOI not confirmed |
| A | `Lanvin2023CICIDSFaulty` | VERIFY | Authors Lanvin et al.; venue likely 2023 security conference; DOI not confirmed |
| A | `ProjectAgentContext`, `ProjectResultsSnapshot`, `Arp2020DosDontsMLSecurity` | Confirmed | Use as-is |
| B | `Sarhan2022StandardFeatureSet` | VERIFY | Authors Sarhan et al.; year approx. 2022; venue and DOI not confirmed |
| B | `ProjectAgentContext` | Confirmed | Use as-is |
| C | `LopezMartin2020DRLIDS` | CANDIDATE | Expert Systems with Applications, 2020; DOI 10.1016/j.eswa.2019.112963 — verify before adding to bib |
| C | `LopezMartin2021RBFOfflineRL` | CANDIDATE | IEEE Access, 2021; DOI 10.1109/ACCESS.2021.3127689 — verify before adding |
| C | `Ren2022IDRDRL` | CANDIDATE | Scientific Reports, 2022; DOI 10.1038/s41598-022-19366-3 — verify before adding |
| C | `Yang2024DRLNIDSSurvey`, `Gueriani2024DRLIoTIDSSurvey` | Confirmed (arXiv) | Audit publication status before submission |
| C | `Dabney2018QRDQN`, `Bellemare2017DistributionalRL` | Confirmed | Use as-is |
| C | ACD survey | VERIFY — not in bib | Add a specific verified ACD paper before including the ACD paragraph |
| D | `LopezMartin2020DRLIDS` | CANDIDATE | See Section C |
| D | `GymnasiumDocs`, `Lee2002CostSensitiveIDS`, `ProjectAgentContext`, `ProjectResultsSnapshot` | Confirmed | Use as-is |
| E | `DiMonda2024FewShotNIDS` | VERIFY | Authors Di Monda et al.; year approx. 2024; venue and DOI not confirmed |
| E | `Yang2024DRLNIDSSurvey`, `ProjectAgentContext` | Confirmed | Use as-is |
| F | `SommerPaxson2010ClosedWorld` | VERIFY | Robin Sommer and Vern Paxson; IEEE S&P 2010; likely DOI 10.1109/SP.2010.25 — confirm and add to bib |
| F | `Axelsson1999BaseRate` | VERIFY | Stefan Axelsson; ACM TISSEC vol. 3, no. 3, 2000; likely DOI 10.1145/357802.357804 — confirm and add to bib |
| F | `Arp2020DosDontsMLSecurity`, `Apruzzese2022CrossEvaluationNIDS`, `Lee2002CostSensitiveIDS`, `ProjectAgentContext` | Confirmed | Use as-is |

---

## Administrative: Citation Key Mismatches in Existing Draft

**Blocking for LaTeX compilation.** Before any TeX build, correct these mismatches in
`report/drafts/state_of_the_art.md`:

| Key used in draft | Problem | Safe replacement |
|---|---|---|
| `Ring2017FlowBasedIDS` | Not in bib | `Sperotto2010FlowIDS` (flow IDS background) or `Ring2019DatasetSurvey` (dataset survey content) — check context per occurrence |
| `CrossDomain2023NIDS` | Not in bib | `Layeghy2023CrossDomainNIDS` |
| `EvalLongTerm2022NIDS` | Not in bib | `Apruzzese2022CrossEvaluationNIDS` (verify content fit) |
| `DatasetSurvey2025NIDS` | Not in bib | Verify publication and add entry; or mark as [citation needed] in draft |
| `DLNIDSSurvey2024` | Not in bib | Verify and add entry; or use `LiuLang2019IDSSurvey` for survey context |
| `CostSensitiveIDSModel` | Not in bib | Replace with `Lee2002CostSensitiveIDS` |
| `CSEIDS2021CostSensitive` | Not confirmed | Verify; remove if not found |

---

## Cumulative Citation Status

| Key | Status | Use |
|---|---|---|
| `Sharafaldin2018CICIDS2017` | Confirmed | CICIDS2017 dataset description |
| `Lashkari2017CICFlowMeter` | Confirmed | CICFlowMeter and flow features |
| `MoustafaSlay2015UNSWNB15` | Confirmed | UNSW-NB15 and cyber-range context |
| `Tavallaee2009NSLKDD` | Confirmed | NSL-KDD historical context |
| `ScarfoneMell2007` | Confirmed | NIDS/IPS definition |
| `SuttonBarto2018RL` | Confirmed | RL fundamentals |
| `Mnih2015DQN` | Confirmed | DQN background |
| `VanHasselt2016DoubleDQN` | Confirmed | Double DQN background |
| `Wang2016DuelingDQN` | Confirmed | Dueling DQN (if mentioned) |
| `Bellemare2017DistributionalRL` | Confirmed | Distributional RL / C51 |
| `Dabney2018QRDQN` | Confirmed | QR-DQN algorithm |
| `GymnasiumDocs` | Confirmed | Gymnasium environment API; pin version |
| `SB3ContribQRDQNDocs` | Confirmed | SB3-Contrib QRDQN; pin version |
| `Lee2002CostSensitiveIDS` | Confirmed | Asymmetric IDS cost; note 2002 vintage |
| `Arp2020DosDontsMLSecurity` | Confirmed (arXiv) | ML security pitfalls |
| `Sperotto2010FlowIDS` | Confirmed | Flow-based IDS overview |
| `Ring2019DatasetSurvey` | Confirmed | NIDS dataset survey |
| `Apruzzese2022CrossEvaluationNIDS` | Confirmed | Cross-domain evaluation |
| `Layeghy2023CrossDomainNIDS` | Confirmed | Cross-domain evaluation |
| `Yang2024DRLNIDSSurvey` | Confirmed (arXiv) | DRL-IDS survey; audit venue |
| `Gueriani2024DRLIoTIDSSurvey` | Confirmed (arXiv) | DRL IoT-IDS survey; audit venue |
| `KDDCup1999` | Confirmed | Historical context only |
| `Koroniotis2019BotIoT` | Confirmed | IoT datasets / future work |
| `Alsaedi2020ToNIoT` | Confirmed | IoT datasets / future work |
| `LiuLang2019IDSSurvey` | Confirmed | ML/DL IDS survey context |
| `ProjectAgentContext` | Confirmed | All project-specific claims |
| `ProjectResultsSnapshot` | Confirmed | Artifact-backed results; always cite with run ID |
| `LopezMartin2020DRLIDS` | CANDIDATE — DOI 10.1016/j.eswa.2019.112963 | Add to bib after verifying |
| `LopezMartin2021RBFOfflineRL` | CANDIDATE — DOI 10.1109/ACCESS.2021.3127689 | Add to bib after verifying |
| `Ren2022IDRDRL` | CANDIDATE — DOI 10.1038/s41598-022-19366-3 | Add to bib after verifying |
| `Engelen2021CICIDSIssues` | VERIFY | Locate venue/DOI; add to bib |
| `Lanvin2023CICIDSFaulty` | VERIFY | Locate venue/DOI; add to bib |
| `Sarhan2022StandardFeatureSet` | VERIFY | Locate venue/DOI; add to bib |
| `DiMonda2024FewShotNIDS` | VERIFY | Locate venue/DOI; add to bib |
| `SommerPaxson2010ClosedWorld` | VERIFY — likely DOI 10.1109/SP.2010.25 | Confirm and add to bib |
| `Axelsson1999BaseRate` | VERIFY — likely DOI 10.1145/357802.357804 | Confirm and add to bib |
| `Cantone2024CrossDataset` | VERIFY | Locate venue/DOI; if unverifiable, use Layeghy/Apruzzese |
| `DatasetSurvey2025NIDS` | VERIFY — not in bib | Verify or remove from draft |
| `DLNIDSSurvey2024` | VERIFY — not in bib | Verify or remove from draft |
| `EvalLongTerm2022NIDS` | KEY MISMATCH | Replace with `Apruzzese2022CrossEvaluationNIDS` |
| `CostSensitiveIDSModel` | KEY MISMATCH | Replace with `Lee2002CostSensitiveIDS` |
| `Ring2017FlowBasedIDS` | KEY MISMATCH | Replace with `Sperotto2010FlowIDS` or `Ring2019DatasetSurvey` |
| `CrossDomain2023NIDS` | KEY MISMATCH | Replace with `Layeghy2023CrossDomainNIDS` |
