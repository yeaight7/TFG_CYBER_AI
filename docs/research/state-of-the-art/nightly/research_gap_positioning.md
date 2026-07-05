# Research Gap and Thesis Positioning
## Second revised pass — 2026-05-16

This document provides a defensible research-gap statement and positioning for the thesis. All
content is in English and intended for human review before adaptation into the chapter draft.
Do not paste any version verbatim without reading it first.

**Citation discipline.** Confirmed bib keys are used directly. Keys marked [VERIFY] require
bibliographic confirmation before final use. No metadata is invented.

---

## 1. The Literature Landscape

Three areas are thoroughly covered and should not be presented as gaps.

**Supervised ML and DL for NIDS** is a mature field. Systematic reviews document many model
families across public datasets; CICIDS2017 was designed specifically as a flow-based benchmark
for this kind of work [Sharafaldin2018CICIDS2017]; and Random Forest and gradient-boosted trees
are established strong baselines on tabular flow data [Sharafaldin2018CICIDSAnalysis]. The thesis
cannot claim supervised NIDS is unsolved.

**RL and DRL for IDS** is an active and already-populated area. Several works have explicitly
treated labelled-dataset records as RL environments, with class decisions as actions and reward
derived from classification accuracy [LopezMartin2020DRLIDS CANDIDATE; Yang2024DRLNIDSSurvey;
Gueriani2024DRLIoTIDSSurvey]. DQN-style agents have been applied to CICIDS2017-family data.
The thesis cannot claim to introduce RL to intrusion detection.

**Flow-based traffic representation** is decades-old [Sperotto2010FlowIDS; Lashkari2017CICFlowMeter].
It does not require novelty justification; it requires a methodological rationale.

---

## 2. What Is Partially Covered and Leaves Room for Contribution

These areas have received attention but leave substantive gaps the thesis can legitimately address.

**Binary PERMIT/BLOCK framing with explicit asymmetric cost.** Most RL-IDS works frame the action
as a class label with a symmetric correct/incorrect reward or leave the reward function unspecified
[Yang2024DRLNIDSSurvey]. The framing used in this thesis — a binary PERMIT/BLOCK decision
vocabulary with a reward that penalises false negatives more heavily than false positives — connects
the RL formulation to the operational semantics of access control. This is defensible as a
methodological contribution, not an algorithmic one.

**Distributional RL in NIDS.** Distributional RL methods (C51, QR-DQN, IQN) do not appear in
the RL-IDS literature reviewed in this thesis [Yang2024DRLNIDSSurvey; Gueriani2024DRLIoTIDSSurvey].
QR-DQN is an exploratory algorithmic choice: its distributional view of returns is conceptually
relevant to a cost-sensitive setting, but this thesis does not claim it is superior. The comparison
against a DQN baseline and against a Random Forest baseline is what would support or refute such
a claim.

**Controlled data-efficiency experiments.** RL-IDS works typically train on full datasets without
controlled variation of training volume [Yang2024DRLNIDSSurvey]. Supervised few-shot NIDS work
exists but is not the same as a controlled RL-vs-supervised scaling comparison at fixed training
sizes. The 100k / 250k / 500k / 1M / 2M experiments with a fixed test set fill this gap at the
methodological level.

**Reproducibility and protocol documentation.** A recurring weakness in RL-IDS literature is the
absence of reported split strategy, preprocessing details, baseline comparisons, or random seed
counts [Yang2024DRLNIDSSurvey; Arp2020DosDontsMLSecurity]. This thesis claims methodological
strength in its explicit anti-leakage policy, staged evaluation ladder, and separation of internal
from external results.

---

## 3. What Is Weakly Validated in the Literature

These known limitations of existing NIDS work give the thesis room to differentiate itself through
methodological rigour rather than algorithmic novelty.

**Benchmark quality.** CICIDS2017 has documented quality issues including mislabelled flows,
incorrect flow boundaries, and a high proportion of artefact flows [VERIFY: Engelen2021CICIDSIssues;
VERIFY: Lanvin2023CICIDSFaulty]. Many studies use the dataset without explicitly acknowledging
these limitations. This thesis addresses them through its curated adapter and anti-leakage policy.

**Evaluation leakage and split design.** Random row-wise splits can inflate NIDS results by placing
related flows in both training and test sets [Arp2020DosDontsMLSecurity; Layeghy2023CrossDomainNIDS].
The thesis addresses this through Check A (random), Check B (shuffled-label validation), Check C
(CSV/day split), and code for leave-one-exact-CSV-out evaluation.

**Cross-domain generalisation.** Models that perform well within one public NIDS dataset degrade
substantially when evaluated on a different dataset or traffic distribution
[Apruzzese2022CrossEvaluationNIDS; Layeghy2023CrossDomainNIDS]. Few RL-IDS papers report
cross-dataset or external-traffic results. The Phase 2 offline-inference design addresses this,
at least in part, by evaluating on lab-captured traffic.

---

## 4. Defensible Research Gap

The thesis gap is **methodological and experimental**, not algorithmic. Use one of the three
versions below; do not blend them into a single statement.

### Version A — Conservative (recommended for bachelor thesis)

This thesis evaluates a reproducible, carefully scoped binary flow-level PERMIT/BLOCK formulation
using a QRDQN agent. The contribution is not algorithmic novelty: RL for IDS already exists, and
CICIDS2017 is a common benchmark. The contribution is an experimental protocol that combines a
fixed canonical feature schema, a Gymnasium-compatible dataset-as-environment, a cost-sensitive
asymmetric reward, leakage-aware staged evaluation, direct comparison against a supervised
baseline, and controlled data-efficiency analysis. The research question is not whether RL can
work for IDS in principle, but whether this specific QRDQN formulation produces results comparable
to a supervised baseline under a reproducible and methodologically cautious protocol.

### Version B — Balanced

This thesis is positioned in the methodological space between two existing traditions: RL applied
as a classifier over labelled NIDS datasets, and RL applied to autonomous cyber defence in
simulated networks. It proposes an offline, flow-level, defender-centric binary PERMIT/BLOCK
formulation with a cost-sensitive reward that explicitly penalises missed attacks more than false
alarms. The formulation is evaluated under a controlled pipeline where training volume is
systematically varied, a supervised baseline is compared under the same conditions, and evaluation
progresses from random splits through harder file-based splits and, where viable, offline inference
on lab-captured traffic. The contribution lies in the protocol design, the careful treatment of
evaluation risk, and the explicit characterisation of what the RL formulation gains and loses
relative to supervised alternatives.

### Version C — Ambitious but defensible

This thesis frames its contribution as a reproducible evaluation protocol for RL-based binary
flow-level network defence that bridges the gap between supervised NIDS benchmarking and autonomous
cyber-defence research. By combining a canonical feature representation, a Gymnasium-based
dataset-as-environment, QRDQN as a distributional RL agent, asymmetric cost-sensitive reward,
controlled data-efficiency experiments, leakage-aware evaluation, and conditional external
validation, the thesis provides structured evidence for when and how a binary PERMIT/BLOCK defender
based on RL compares to tabular supervised models under controlled conditions. The work does not
claim production readiness, real-time blocking, or algorithmic superiority. It provides a
transparent experimental baseline for future research that may extend the formulation toward online
environments or operational deployment.

**Recommendation.** Use Version A in the introduction and memory chapter. Version B is appropriate
for the positioning paragraph at the end of the State of the Art chapter. Version C should be
reserved for cases where the experiment results fully support the additional framing — do not use
it pre-emptively.

---

## 5. Claims That Must Not Appear in the Thesis

| Forbidden claim | Why it would be challenged | Safe alternative |
|---|---|---|
| "RL for IDS has not been studied" | Directly contradicted by multiple peer-reviewed works | "RL for IDS is an active area; this thesis evaluates a specific formulation" |
| "This is the first dataset-as-environment IDS" | López-Martín 2020 and others do this independently | "The dataset-as-environment design follows a pattern identified in the literature" |
| "QRDQN is proven superior for NIDS" | No same-protocol comparison against DQN exists in this work | "QRDQN is evaluated as a distributional RL candidate under this protocol" |
| "The system performs real-time blocking" | Phase 2 is offline inference; no inline blocking implemented | "The implementation performs offline decision inference over extracted flow records" |
| "CICIDS2017 performance proves generalisation" | Cross-domain studies show intra-dataset metrics do not transfer | "CICIDS2017 provides a reproducible internal benchmark; generalisation is a separate question" |
| "External validation is complete" | No committed Phase 2 artifact covers attack traffic | "Phase 2 provides a preliminary false-positive check on benign lab traffic" |
| "The reward function captures real operational cost" | Reward weights are scenario assumptions, not measured | "The reward operationalises an asymmetric cost preference consistent with cost-sensitive IDS literature" |
| "This project is production-ready" | Experimental prototype; no deployment context | "This is a scoped experimental study" |
| "CICIDS2017 is a realistic representation of current traffic" | Traffic is from 2017, lab-generated, with known quality issues | "CICIDS2017 provides a widely used and reproducible benchmark, with the limitations documented in this thesis" |

---

## 6. Evidence Mapping

| Gap claim | Supporting sources | Status |
|---|---|---|
| RL-IDS already exists; cannot claim novelty | Yang2024DRLNIDSSurvey; Gueriani2024DRLIoTIDSSurvey; LopezMartin2020DRLIDS | Confirmed / CANDIDATE |
| Distributional RL (QRDQN) not in reviewed NIDS literature | Yang2024DRLNIDSSurvey; Dabney2018QRDQN | Confirmed |
| Binary PERMIT/BLOCK with explicit FN cost is less common | Yang2024DRLNIDSSurvey; Lee2002CostSensitiveIDS | Confirmed |
| Controlled data-efficiency experiments rare in RL-IDS | Yang2024DRLNIDSSurvey | Confirmed |
| CICIDS2017 has documented quality issues | Engelen2021CICIDSIssues; Lanvin2023CICIDSFaulty | VERIFY |
| Cross-domain generalisation is poor | Apruzzese2022CrossEvaluationNIDS; Layeghy2023CrossDomainNIDS | Confirmed |
| Random splits can inflate metrics | Arp2020DosDontsMLSecurity; Layeghy2023CrossDomainNIDS | Confirmed |
| Supervised baselines are strong on tabular flow data | Sharafaldin2018CICIDSAnalysis | Confirmed partial |
| Benchmark performance ≠ operational readiness | SommerPaxson2010ClosedWorld; Arp2020DosDontsMLSecurity | VERIFY / Confirmed |

---

## 7. Thesis-Ready Sentences

These sentences are drafted in academic register and may be used after human review.

**Opening positioning sentence** (Section 12 of SoA, or introduction chapter):
> This thesis does not claim to introduce reinforcement learning to intrusion detection, nor to
> demonstrate that QRDQN is optimal for network security. It studies a specific, scoped
> experimental question: whether a binary flow-level PERMIT/BLOCK formulation, using a QRDQN
> agent trained over a canonical feature schema derived from CICIDS2017, can be evaluated under
> a reproducible and leakage-aware protocol that supports honest comparison with a supervised
> baseline.

**Transition sentence after related work:**
> The pattern across this literature is that evaluation rigour varies substantially. Studies often
> report high intra-dataset accuracy without temporal splits, explicit leakage controls, supervised
> baseline comparison, or external validation. This thesis is positioned to address that gap at
> the methodological level, not by claiming algorithmic novelty.

**Closing sentence for positioning section:**
> The value of this work lies not in introducing reinforcement learning to intrusion detection,
> but in providing a structured, documented, and reproducible experimental protocol that
> characterises what a binary QRDQN defender gains and loses relative to supervised alternatives
> under controlled conditions on a standard public benchmark.
