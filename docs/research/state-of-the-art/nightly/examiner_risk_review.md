# Examiner Risk Review
## Nightly agent pass — 2026-05-16

This document anticipates the questions and objections a thesis examiner is likely to raise, and
drafts honest, defensible responses for each. It is intended for the thesis author to read before
the defence, and for the thesis writing to pre-empt the most predictable challenges.

Tone: direct and honest. Do not try to hide the weaknesses; name them, bound them, and explain
what the thesis does in response. An examiner is more persuaded by honest scoping than by
defensive overclaiming.

---

## 1. Core Examiner Challenges

### 1.1 "Why use RL instead of a supervised classifier?"

**The challenge.** Supervised learning — especially Random Forest and gradient-boosted trees — is
well-established for tabular flow classification and frequently achieves high performance on
CICIDS2017. An examiner will ask why the additional complexity of RL is justified.

**Honest response.** It may not be justified in terms of raw classification performance, and the
thesis does not claim that it is. The RL formulation is studied for two reasons. First, a reward
function can encode asymmetric false-positive and false-negative costs directly during training,
without requiring post-hoc calibration of a supervised loss. Second, the PERMIT/BLOCK action
vocabulary is a natural interface for a decision-based defender that could, in principle, be
extended to an online or adaptive setting. Whether the RL approach is competitive with Random
Forest on this specific dataset is precisely what the experimental comparison is designed to
determine. If the Random Forest matches or outperforms QRDQN, that is a valid and useful result,
not a failure of the thesis.

**Pre-emptive action.** Include the Random Forest comparison with full metrics. State explicitly
in the methodology and conclusion that the purpose of the comparison is to determine whether the
additional complexity of RL is warranted, and report the answer honestly regardless of the outcome.

---

### 1.2 "Is CICIDS2017 still representative of modern traffic?"

**The challenge.** CICIDS2017 was captured in 2017 using specific attack scripts, a controlled
lab network, and CICFlowMeter feature extraction. An examiner may argue that 2017 traffic patterns
and attack vectors are no longer representative of current threats.

**Honest response.** CICIDS2017 is not claimed to represent current operational traffic. It is
used as a reproducible benchmark: the thesis evaluates whether the proposed formulation works
under controlled conditions, not whether it generalises to all current network environments. The
dataset is widely used in the research community precisely because it provides a shared evaluation
surface, and the thesis contributes to that shared context. The limitations of CICIDS2017 are
explicitly acknowledged, including its controlled lab origin and known quality concerns.

**Pre-emptive action.** Devote a paragraph to CICIDS2017 limitations in Section 4 of the SoA.
Cite the quality concerns (Engelen, Lanvin) once verified. Separate internal benchmark performance
from any external validation claim.

---

### 1.3 "Why QRDQN specifically? What is the justification over standard DQN?"

**The challenge.** QRDQN has strong RL benchmark evidence but no NIDS-specific track record. An
examiner may ask why the distributional variant was chosen rather than a simpler DQN or PPO agent.

**Honest response.** QRDQN models a distribution over returns rather than only the expected return.
In a cost-sensitive setting where false negatives and false positives carry different weights,
the tail of the return distribution carries decision-relevant information, and a distributional
agent can in principle be made aware of that tail. This is a conceptual motivation, not a
demonstrated performance advantage: the thesis does not claim QRDQN outperforms DQN on NIDS.
The choice is framed as an exploratory algorithmic decision. A DQN baseline comparison would
strengthen the claim that distributional RL adds value in this context.

**Pre-emptive action.** State this honestly in the algorithm chapter. If a DQN baseline is not
included in the experiments, note that comparing QRDQN against DQN is a natural direction for
future work.

---

### 1.4 "The dataset-as-environment is just classification with an RL wrapper. Why is it RL?"

**The challenge.** An examiner may argue that if each observation is an independent labelled row
and reward is derived from the label, the formulation is equivalent to a custom cost-sensitive
classifier with extra terminology and added complexity.

**Honest response.** This objection is partly correct and should be acknowledged directly. The
formulation is closer to reward-engineered classification than to rich sequential RL, because
the environment has no state transition dynamics: each observation is independent, the action
does not influence future observations, and the "environment" is essentially a data loader with
a reward function. The value of the RL interface is twofold: it makes the cost structure explicit
during training rather than post-hoc, and it creates a Gymnasium-compatible prototype that can
be extended to include temporal context, sequential state, or adversarial dynamics in future work.
The thesis does not claim the current formulation provides richer sequential modelling than a
supervised classifier.

**Pre-emptive action.** Address this in Section 9 of the SoA (classification-as-RL tension).
Keep the paragraph on the critical argument; do not soften it.

---

### 1.5 "Can PERMIT/BLOCK decisions be implemented in a real network?"

**The challenge.** An examiner may ask whether the PERMIT/BLOCK decisions produced by the model
can actually be used to control a network — and note that the thesis implies a firewall-like
capability without implementing one.

**Honest response.** The current implementation does not block any traffic. Phase 2 is offline
inference over extracted flow CSVs; there is no inline packet dropping, no firewall rule update,
and no real-time action. The PERMIT/BLOCK labels are decision outputs used for evaluation purposes.
The vocabulary is adopted because it is operationally meaningful and connects the formulation to
a realistic future deployment scenario, not because the scenario is implemented. Operational
deployment would require integration with a network control plane and additional engineering that
is outside the scope of this thesis.

**Pre-emptive action.** State this limitation explicitly in the introduction, in Section 1 of the
SoA, and in the conclusion. The phrase "offline decision inference" should appear consistently
throughout. Never use the phrase "real-time blocking" in the thesis.

---

### 1.6 "Your external validation is benign-only — that is not real external validation."

**The challenge.** An examiner may note that a benign-only Phase 2 lab capture cannot test whether
the model detects attacks that differ from CICIDS2017 attack patterns. It only tests whether the
model produces false positives on benign traffic from a different source.

**Honest response.** This is correct. A benign-only external capture is a false-positive and
domain-shift sanity check, not a complete external validation. It provides useful information
about whether the model over-blocks benign traffic from a different network environment, but it
does not establish detection performance on unseen attacks. The thesis should describe Phase 2
as a preliminary distribution-shift check, not as external validation in the strong sense.

**Pre-emptive action.** Use conditional language throughout: "if lab attack traffic is available,
the external validation will cover both benign and attack flows; otherwise it will serve as a
false-positive check." Describe the current Phase 2 artifacts with their exact scope: benign-only,
specific run IDs, specific conditions.

---

### 1.7 "Why those specific training sizes (100k / 250k / 500k / 1M / 2M)?"

**The challenge.** An examiner may ask whether the training sizes are arbitrary or grounded in
some principled argument about the dataset size and the task.

**Honest response.** The sizes are chosen to span roughly three orders of magnitude within the
available CICIDS2017 data volume, at logarithmically spaced increments. They are not derived from
a theoretical model; they are a practical choice that allows characterisation of the performance
curve across a range from data-sparse to data-rich settings. The intent is to observe whether
RL performance converges toward supervised performance as data grows, or whether there is a
crossing point or consistent gap.

**Pre-emptive action.** State the rationale for the size choices in the methodology section.
Include the total available training data count so the reader can interpret the sizes in context.

---

## 2. Methodological Weaknesses

| Weakness | Severity | Honest framing |
|---|---|---|
| RF baseline metrics not yet generated | High | "The Random Forest baseline protocol is implemented; results are pending and will be added before final submission." Do not report QRDQN results without the baseline comparison. |
| No multi-seed RL experiments | Medium | "Results are reported for single training runs; robustness across seeds is a limitation and a direction for future work." |
| leave-one-CSV-out artifact not committed | Medium | "Code for leave-one-CSV-out validation exists; a full committed artifact is pending." Do not report metrics that have not been generated. |
| Reward sensitivity not tested | Medium | "The reward weights are a scenario assumption. Sensitivity analysis is a direction for future work." |
| No DQN-vs-QRDQN comparison | Low-Medium | "The distributional advantage of QRDQN over plain DQN is not directly measured in this thesis; it is left for future work." |
| Citation key mismatches | Blocking | Fix before any TeX build. See `state_of_art_expansion.md` administrative section. |

---

## 3. Dataset Limitations

| Limitation | How to frame it honestly |
|---|---|
| CICIDS2017 is from 2017 | "CICIDS2017 provides a reproducible benchmark widely used in the research community. Its traffic and attack patterns reflect 2017 conditions and do not represent current threats. Performance on this dataset is reported as internal benchmark evidence, not as a claim about modern operational environments." |
| Lab-generated, not ISP traffic | "The dataset was generated in a controlled CIC network laboratory using traffic generators and attack scripts. This provides clean labels and reproducible conditions, but the traffic distribution does not represent production ISP or enterprise traffic." |
| Known quality issues | "Prior work has identified flow-boundary errors, mislabelled flows, and a high proportion of artefact records in CICIDS2017. This thesis addresses these concerns through a curated adapter and anti-leakage policy; it does not guarantee that all artefacts are eliminated." |
| Class imbalance | "CICIDS2017 is class-imbalanced, with benign traffic dominating some daily files. This is addressed through per-class evaluation metrics (precision, recall, F1, FPR, FNR) and reported in the confusion matrix." |
| No cross-dataset evaluation in internal benchmark | "Internal evaluation uses CICIDS2017 only. Cross-dataset performance is addressed through Phase 2 lab-traffic inference, which is a separate evaluation stage." |

---

## 4. RL vs. Supervised Learning Objections

| Objection | Honest response |
|---|---|
| "RL is more complex and less interpretable than Random Forest" | Correct. The thesis does not claim interpretability as a benefit. The intended benefit is explicit cost encoding during training. Interpretability is listed as a limitation. |
| "Random Forest often matches DL on tabular data; why would RL be different?" | It may not be. The comparison against Random Forest is the experimental test of this question. If RF is competitive, the thesis reports that result and discusses its implications. |
| "The reward function is an engineering choice, not a principled learning signal" | The reward weights are scenario assumptions, not derived from data. This is acknowledged. The thesis claims that encoding asymmetric cost in the reward function is methodologically coherent with the cost-sensitive IDS literature, not that the specific weights are optimal. |
| "RL requires more data and longer training than supervised alternatives" | This is likely true, and the data-efficiency experiments are specifically designed to characterise this. The learning curve experiments will show how QRDQN performance compares to RF at each training budget. |

---

## 5. CICIDS2017 Realism and Staleness Concerns

The most predictable examiner objection about the dataset is not about quality but about age and
realism. Here is the safest thesis framing:

> CICIDS2017 is used as the primary internal benchmark because it provides labelled, reproducible
> flow-based traffic with a documented feature extraction pipeline. Its use here is consistent with
> its role in a large body of NIDS research published between 2017 and 2024. The thesis does not
> claim that performance on CICIDS2017 implies operational effectiveness against current attack
> vectors. CICIDS2017 is treated as the controlled evaluation surface against which the proposed
> formulation is characterised; generalisability to current traffic is a separate question addressed
> through external validation and discussed explicitly as a limitation.

---

## 6. Why Binary PERMIT/BLOCK Is a Valid Simplification

An examiner may question whether reducing NIDS to a binary decision is oversimplified. The
defence has three parts:

**Part 1: Operational precedent.** Network access control decisions are fundamentally binary in
many deployment contexts: a firewall rule either allows or denies a flow. The PERMIT/BLOCK
vocabulary maps naturally to this framing.

**Part 2: Explicit asymmetry.** A binary formulation forces the explicit choice about what to
do with each flow, and the reward function makes the relative cost of each error type explicit.
This is arguably cleaner than a multi-class formulation where the access control consequence of
each class prediction must be inferred separately.

**Part 3: Honest scope.** The thesis is not claiming that binary classification solves intrusion
detection. It is studying a binary decision agent as a specific, scoped experimental prototype.
The limitations of the binary formulation — no attack-family discrimination, no confidence
scores, no multi-step response — are acknowledged in the limitations section.

---

## 7. How to Defend the Thesis Contribution

The safest and most persuasive defence strategy is to be explicit that the contribution is
**methodological and experimental**, not algorithmic or operational.

The thesis contributes:
1. A reproducible binary flow-level PERMIT/BLOCK formulation with a documented canonical schema.
2. A Gymnasium-compatible dataset-as-environment with cost-sensitive reward design.
3. A staged evaluation ladder: random split (Check A), shuffled-label validation (Check B),
   CSV/day split (Check C), leave-one-CSV-out (code exists; artifact pending).
4. A direct supervised baseline comparison (Random Forest; results pending).
5. Controlled data-efficiency experiments across five training budgets.
6. Explicit separation of internal benchmark performance from external lab-traffic inference.

These contributions do not require QRDQN to be better than Random Forest. They require honest
reporting of what the experiments show, careful treatment of evaluation risk, and explicit
acknowledgement of what the thesis does not claim.

The most persuasive closing statement at a defence is approximately:

> "This thesis does not claim that reinforcement learning is superior to supervised learning for
> intrusion detection. It claims that a binary PERMIT/BLOCK RL formulation can be evaluated under
> a reproducible, leakage-aware protocol that is more methodologically careful than most of the
> existing RL-IDS literature. Whether the RL formulation is worth the additional complexity is
> what the experimental results show, and the thesis reports that result honestly."
