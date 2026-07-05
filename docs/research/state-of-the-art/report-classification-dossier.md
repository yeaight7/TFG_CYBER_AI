<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Classification-as-RL Dossier

## 1. Conceptual explanation

**Claim 1 (basic formulation).** A static classification problem can be cast as a trivial Markov Decision Process (MDP) or contextual bandit where each data sample is a state/observation $x$, each possible class label is an action $a$, the environment returns a scalar reward $r(x,a)$ based on the correctness or cost of the prediction, and episodes are short sequences of such decisions (often of length 1 or a fixed number of samples).[RLBook2018; FairRLClass2024 — evidence: strong; caveat: this MDP has no meaningful dynamics beyond sampling from the dataset; use: to formally justify your Gym-style dataset environment][^1][^2]

- **State $s$**: feature vector of one sample (in your case, one flow), e.g. $s = \phi(\text{flow})$.[ZeroDayRL-Protocol — strong as direct example][^3]
- **Action $a$**: predicted class or decision (e.g., PERMIT/BLOCK).
- **Reward $r$**: scalar based on correctness or cost; e.g., +1 for correct label, −1 for incorrect, or more generally a cost‑sensitive reward matrix.[RLCC2023; ZeroDayRL-Protocol — strong][^4][^3]
- **Transition $P(s'|s,a)$**: often independent re‑sampling of another data point; in many implementations there is no dependence on $s,a$, so the MDP reduces to a contextual bandit.[FairRLClass2024 — strong][^2]
- **Episode**: a finite sequence of classifications (e.g., one full pass over a minibatch or over the whole training set), after which cumulative reward is computed and the environment resets.[ZeroDayRL-Protocol — moderate][^3]
- **Policy $\pi_\theta(a|s)$**: mapping from sample features to a distribution over class labels, implemented by a neural network in DQN/QRDQN or actor‑critic methods.[RLCC2023; DQLIDS2021 — strong][^4][^5]

**Claim 2 (relation to supervised learning).** When each state is an independent sample from a fixed dataset and rewards are deterministic functions of the true label (e.g., $r=1$ if $a=y$, $r=0$ otherwise), maximizing expected return is equivalent to minimizing classification error, so the RL formulation reduces mathematically to cost‑sensitive classification or contextual bandits.[RLBook2018; ContextBanditsCost2023; RLCC2023 — evidence: strong; caveat: RL algorithms may still behave differently due to exploration and bootstrapping; use: to honestly explain that you are not “discovering” a new problem class][^6][^1][^4]

Your flow‑based NIDS setting fits this pattern:

- **State**: cleaned flow feature vector.
- **Action**: $a\in\{0,1\}$ corresponding to PERMIT/BLOCK.
- **Reward**: immediate scalar reward shaped from FP/FN costs.
- **Episode**: e.g., sequence of N random flows; environment resets at end of sequence.
- **Policy**: QRDQN approximates $Z(s,a)$, the return distribution, which collapses to a distribution over the (cost‑sensitive) instantaneous reward plus any shaping you add.

This makes your environment a **dataset‑as‑environment contextual bandit with cost‑sensitive binary actions**, implemented on top of Gymnasium.

***

## 2. Source matrix

Key works that explicitly or implicitly use “classification as RL” formulations.


| Citation key | Paper / source | Year | Domain | State definition | Action definition | Reward definition | Dataset | Algorithm | Baselines | Main contribution | Criticism / limitation | Relevance to my thesis |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| RLBook2018 | Sutton \& Barto, *Reinforcement Learning: An Introduction* | 2018 | RL theory | Abstract; MDP state $s$ | Action $a$ in discrete/continuous space | Cumulative scalar reward, often immediate $r(s,a)$ | Conceptual | Tabular \& function-approx. RL | Supervised learning, bandits | Formalizes MDPs, contextual bandits, and relation to supervised learning in one‑step problems | Does not discuss classification as RL explicitly; general theory | Provides theoretical frame: your environment is essentially a contextual bandit with cost‑sensitive loss[^1][^7] |
| RLCC2023 | Zhang et al., “Reinforcement learning-based cost-sensitive classifier for imbalanced fault classification,” *Science China Information Sciences* | 2023 | Fault diagnosis (imbalanced classification) | Feature vector of an instance | Action = class label from set of fault types | Reward derived from misclassification cost matrix; policy gradient learns to minimize expected cost | Fault classification datasets (industrial) | Actor‑critic RLCC (policy gradient + critic) | Cost‑sensitive SVM, cost‑sensitive NN, standard classifiers | Shows RL can directly optimize a non‑uniform cost matrix and adapt sample weights for imbalanced classification[^4][^8] | Environment is static supervised dataset; RL used mainly as optimization technique; sample inefficiency vs. direct cost‑sensitive training is not fully quantified | Strong conceptual precedent for using RL to encode asymmetric FP/FN costs in static classification; supports your reward design rationale |
| FairRLClass2024 | “Balancing the Scales: Reinforcement Learning for Fair Classification,” arXiv | 2024 | Fair classification | State = features + sensitive attributes | Action = predicted label / decision | Reward encodes accuracy minus fairness penalty; long‑run return trades off performance and fairness | Tabular \& real fair‑classification benchmarks | Policy‑gradient RL | Fairness‑aware supervised methods | Theoretically frames classification as RL to optimize complex objectives (fairness + accuracy) not easily handled by standard losses[^2] | Static i.i.d. setting; RL mostly re‑implements weighted loss; computational overhead vs. benefit depends on problem | Shows RL formulation is useful when optimizing non‑standard, multi‑term objectives (e.g., fairness, cost) — analogous to your cost‑sensitive, risk‑aware reward |
| ContextBanditsCost2023 | “No-Regret Contextual Bandits for Cost-Sensitive Decision-Making,” OpenReview | 2023 | General cost-sensitive decisions | State/context = feature vector | Action = decision among K options | Per‑round loss/cost; goal is low cumulative cost (regret) | Synthetic + civic workflow data | Contextual bandits (EXP4, ILTCB, Online-Cover) | Cost‑sensitive supervised classifiers with retraining | Strong theoretical and empirical evidence that contextual bandits (a one‑step RL formulation) can outperform static classifiers under drift and partial feedback[^9][^6] | Assumes online streaming data and partial feedback; not directly static offline classification | The closest *theoretically clean* analogue to your formulation: per‑sample decision with cost‑sensitive reward; supports your use of RL in view of non‑stationary or future adaptive deployment |
| ZeroDayRL-Protocol | “Preparing dataset for reinforcement learning environment” (DQN for zero‑day vulnerability identification) | 2025 | Security classification (zero-day) | State = feature vector of current data point | Action = {malicious, benign} | Reward = +10 for correct prediction, −1 for incorrect | Security dataset (not named) | DQN | Supervised classifiers (conceptually) | Explicit, detailed description of mapping a binary classifier into a Gym‑like RL environment where each sample is a state and class is action; includes reward definition and episode termination rules[^3] | Protocol‑style writeup, not high‑impact venue; lacks rigorous comparison with supervised baselines | Direct blueprint for your dataset‑as‑environment design; use as **secondary, methodological** citation (clearly marked as such) |
| DQLIDS2021 | “Deep Q-Learning based Reinforcement Learning Approach for Network Intrusion Detection,” preprint | 2021 | NIDS (NSL-KDD) | State = feature vector of a connection (NSL‑KDD record) | Action = discrete class (normal or one of attack types) | Reward based on correct vs incorrect classification; details often: +1/0 or +1/−1 | NSL-KDD | Deep Q-Network | Classical ML classifiers (SVM, RF, etc.) | Shows that a DQN classifier on NSL‑KDD can reach accuracy comparable to traditional ML, and explores hyperparameters such as discount factor and episodes[^5] | Uses outdated dataset; limited discussion of why RL is preferable to standard supervised learning; static environment | Concrete example of NIDS formulated exactly as classification‑as‑RL; relevant to your positioning but must be critically discussed |
| DCIDS2024 | Yu et al., “Deep Q-Network-Based Open-set Intrusion Detection Solution for Industrial Internet of Things,” *IEEE IoT J.* | 2024 | IIoT open‑set NIDS | State = CNN features + confidence metrics (max prob, prob gap, entropy) | Action = accept known class vs label as unknown / suspicious | Reward encourages correct discrimination between known classes and unknown attacks, penalizing misclassified unknowns | TON_IoT (IoT) | DQN (with CVAE‑augmented value network) | Threshold‑based open‑set methods | Casts open‑set recognition as an MDP and uses DQN to learn decision boundaries that better handle unknown attacks[^10] | Environment still built on static offline dataset; sequential aspect is thin; added complexity vs. simpler open‑set thresholds may or may not be justified | Shows RL can be helpful for *open‑set* NIDS decisions; relevant to your **future work** rather than your current closed‑set PERMIT/BLOCK |
| DQNIDS2026 | Tiwari et al., “DQN-IDS: A Deep Reinforcement Learning Approach for Open Set-Enabled Intrusion Detection,” NDSS | 2026 | NIDS (open set, IoT and others) | State = CNN softmax confidence features for each flow | Action = accept as known, reject as unknown | Reward for correct known/unknown decisions, penalizing missed zero‑days and false alarms | CICIDS2017, UNSW‑NB15 | CNN + DQN | Threshold‑based open‑set detectors | Demonstrates that a DQN layer on top of a CNN classifier can reduce missed zero‑day traffic while keeping runtime overhead modest[^11][^12] | RL part acts as post‑hoc decision layer; sequential structure is minimal; evaluation still mostly static | Strong precedent for *combining* supervised models with RL policies in NIDS; useful when arguing for RL as policy layer rather than direct classifier |
| RLDiag2023 | Yu et al., “Deep Reinforcement Learning for Cost-Effective Medical Diagnosis,” arXiv | 2023 | Sequential medical diagnosis | State = patient features + previously ordered tests | Action = which test to order next, or which diagnosis to make | Reward shaped to maximize F1 subject to test cost budget | Clinical datasets | Semi‑model-based deep RL | Supervised baselines, cost‑sensitive decision trees | RL treats diagnosis as sequential decision‑making (which tests to order, when to stop) optimizing non‑standard objective (budget‑constrained F1); RL strictly dominates static classifiers in this decision model[^13] | More sequential than your setting; not pure static classification; requires well‑designed reward shaping and environment model | Useful as **positive example** of when RL is genuinely more appropriate (sequential costs), to contrast with your static flow classification |

*(You probably only need RLCC2023, DQLIDS2021, DCIDS2024/DQNIDS2026, ZeroDayRL-Protocol, ContextBanditsCost2023 and RLDiag2023 in the thesis; keep others as optional.)*

***

## 3. Arguments in favor

For each argument, I’ll mark evidence strength and caveats.

1. **Cost‑sensitive decisions beyond simple cross‑entropy.**
    - RLCC2023 and FairRLClass2024 show that RL/actor‑critic can directly optimize a cost matrix or fairness‑augmented objective, treating per‑sample rewards as functions of prediction and cost.[^8][^2][^4]
    - Evidence strength: **strong**.
    - Caveat: similar behavior can often be obtained with cost‑sensitive losses in supervised learning; RL provides an alternative optimization framework rather than fundamentally new objectives.
    - Use: justify cost‑sensitive reward design (asymmetric penalties for FN vs FP) in your RL environment as consistent with existing work.
2. **Natural bridge to contextual bandits and sequential defense.**
    - Contextual bandit theory shows that learning a per‑context policy with exploration can reduce cumulative cost in non‑stationary settings compared to static classifiers.[^9][^6]
    - Evidence strength: **strong** (theoretical + empirical).
    - Caveat: requires online deployment and partial feedback to fully benefit.
    - Use: argue that your Phase 1 “static dataset as RL environment” is a controlled stepping stone towards future online policies (e.g., adaptive NIDS thresholds, real‑time blocking).
3. **Policy learning rather than scoring function only.**
    - RL can learn policies that consider not only immediate correctness but also long‑term metrics (e.g., long‑run F1 under budget constraints, fairness over time, or defensive posture), as in RL‑based medical diagnosis and fair classification.[^13][^2]
    - Evidence strength: **moderate**, problem‑dependent.
    - Caveat: in your *current* setup, cumulative reward is just sum of per‑sample classification costs; long‑term structure is weak.
    - Use: frame your work as building a **policy‑learning pipeline** that could later be extended to sequential cyber defence (e.g., blocking decisions affecting future traffic or alerts).
4. **Reward shaping for FN/FP asymmetry and risk.**
    - Cost‑sensitive IDS and RLCC show that misclassification costs can be significantly asymmetric, and RL can optimize such objectives.[^14][^15][^4]
    - Distributional RL (QRDQN) allows you to look at the distribution of returns and potentially adopt risk‑averse policies that prioritize avoiding false negatives.[^16][^17]
    - Evidence strength: **moderate–strong**.
    - Caveat: risk‑aware policy selection from return distributions requires additional design (e.g., quantile‑based decision rules), which you should clearly delimit as planned vs implemented.
    - Use: justify the choice of QRDQN as consistent with risk‑sensitive decision‑making under asymmetric costs.
5. **Compatibility with future adaptive defence architectures.**
    - DQN‑IDS and DC‑IDS use DQN as a policy layer on top of learned features to handle open‑set and unknown attacks, showing that RL agents can be integrated into IDS pipelines.[^11][^10]
    - Evidence strength: **moderate** (new but reasonable results).
    - Caveat: their improvements vs. well‑tuned supervised baselines must be carefully interpreted; they are not universal.
    - Use: position your RL formulation as a **minimal instantiation** of an RL‑based defender that could later be extended to open‑set or sequential response settings.

***

## 4. Arguments against

1. **Static classification is already well solved by supervised learning.**
    - For i.i.d. datasets with fully observed labels, minimizing cross‑entropy or cost‑sensitive loss on a supervised classifier is statistically efficient and well understood; RL adds exploration and bootstrapping overhead with no clear benefits when the environment is static and fully known.[RLBook2018; RLCC2023 — strong][^1][^4]
    - In DQLIDS2021 and related NIDS RL papers on NSL‑KDD/CICIDS2017, DQN classifiers often match but rarely dramatically outperform strong supervised baselines while requiring more tuning and compute.[^5][^18]
2. **Reward equals label; no genuine credit assignment.**
    - In most classification‑as‑RL papers, the reward is a deterministic function of the label and chosen class (e.g., +1/−1), so there is no temporal credit assignment or delayed rewards; the problem collapses to per‑sample loss minimization.[ZeroDayRL-Protocol; RLCC2023 — strong][^4][^3]
    - This makes the RL formulation mathematically equivalent to cost‑sensitive classification or contextual bandits, not a truly sequential MDP.
3. **No meaningful dynamics when rows are independent.**
    - If each step’s state is an independent sample from the dataset, the transition kernel is $P(s'|s,a)=P_{\text{data}}(s')$ and does not depend on action, so there is no notion of actions influencing future states — the core feature of RL.[RLBook2018 — strong][^1]
    - Episodes composed of randomly sampled rows are an artificial construction to fit the Gym API rather than reflecting a real temporal process.
4. **Sample inefficiency and optimization overhead.**
    - RL methods like DQN/QRDQN are usually more sample‑inefficient and unstable than supervised training; using them on static datasets without any exploration benefit can be wasteful.[FairRLClass2024; RLDiag2023 — moderate][^2][^13]
    - Without careful baselines and hyperparameter tuning, RL classifiers may underperform simpler supervised models (RF, logistic regression, small MLP).
5. **Risk of unjustified novelty claims.**
    - “Casting classification as RL” by itself is not new; RLCC2023, text classification RL, and earlier works have already explored this view in various domains.[^19][^2][^4]
    - Overstating the novelty of using RL on CICIDS2017 can be misleading; you must frame it as **evaluating QRDQN on a cost‑sensitive binary NIDS task under strict protocols**, not as a fundamentally new problem formulation.

***

## 5. How to defend this formulation in your thesis

**Balanced defense strategy:**

1. **Frame Phase 1 as a controlled RL formulation of a cost-sensitive binary classifier.**
    - Emphasize that Phase 1 uses a static CICIDS2017 dataset environment where each flow is a state, actions are PERMIT/BLOCK, and rewards encode asymmetric FP/FN costs.
    - Cite RLCC2023 and ZeroDayRL‑Protocol as prior art for static classification‑as‑RL, and contextual bandits/cost‑sensitive literature as theoretical grounding.[^6][^3][^4]
    - Explicitly acknowledge that this is **functionally close to cost‑sensitive supervised learning**, and that the contribution lies in comparing QRDQN vs RF under rigorous NIDS evaluation, not in inventing a new RL problem.
2. **Connect Phase 1 to Phase 2 and future sequential defence.**
    - Argue that once you have a policy $\pi(a|s)$ trained in this simplified context, it can be embedded as a decision component in more sequential settings (e.g., a NIDS pipeline where blocking affects future traffic or alert rate).
    - Use DC‑IDS and DQN‑IDS as examples of RL policies on top of supervised features to handle open‑set or IIoT environments, indicating a plausible path from your binary defender to more sophisticated RL‑based defences.[^10][^11]
3. **Maintain strong baseline comparisons.**
    - Commit to always comparing QRDQN to at least one strong supervised baseline (RF, possibly also a simple MLP) on identical features and splits.
    - If RF outperforms or matches QRDQN, state this explicitly; your thesis can still contribute by providing a **negative or nuanced result** under a rigorous evaluation protocol.
4. **Avoid claims that RL is automatically superior.**
    - Phrase RL benefits cautiously: e.g., *“Distributional RL via QRDQN provides a flexible framework for cost-sensitive policies and potential risk‑aware decisions; in this work we evaluate whether those benefits translate into improved NIDS performance on CICIDS2017 under strict evaluation protocols.”*
    - Do not claim that RL “learns from interaction with the real network” — your current setup is offline and dataset‑driven.
5. **Explicitly state when the formulation is artificial.**
    - In your methodological discussion, include a sentence like: *“The episodes in our environment are constructed by sampling flows from a static dataset, so actions do not influence future states; the formulation is therefore closer to a contextual bandit than to a fully sequential MDP.”*
    - Explain that this construction is mainly to reuse well‑tested RL libraries (Gymnasium, SB3‑contrib) and to prepare for future extensions.

***

## 6. Suggested subsection

### Title

**“Formulation of the classification of flows as a reinforcement learning environment”**

### Outline

1. **Motivation for an RL formulation**
2. **MDP / contextual bandit view of flow classification**
3. **Reward design with asymmetric FP/FN costs**
4. **Limitations of the static dataset formulation**
5. **Connection to supervised baselines and future sequential defence**

### Thesis-ready paragraphs (template, with citation keys)

1. **Motivation and high-level formulation**

> In this thesis, we formulate the binary classification of network flows (benign vs. ataque) as a reinforcement learning environment in order to encode explicit cost asymmetries between false positives and false negatives and to reuse standard RL tooling (Gymnasium, SB3‑contrib) for policy learning.[RLBook2018; RLCC2023 — evidencia: fuerte] Each flujo preprocesado se representa como un estado $s$ en un espacio de estados continuo, la acción $a\in\{0,1\}$ corresponde a las decisiones PERMIT/BLOCK, y el entorno devuelve una recompensa escalar $r(s,a)$ que depende de si la decisión coincide con la etiqueta real y del coste asociado a cada tipo de error.[RLCC2023; ZeroDayRL-Protocol — fuerte][^1][^3][^4]

2. **MDP / contextual bandit view**

> Desde el punto de vista teórico, este entorno puede verse como un MDP degenerado o un bandido contextual: los estados son muestras i.i.d. del conjunto de datos, las acciones son etiquetas, la transición $P(s'|s,a)$ no depende de la acción y la recompensa es un funcional determinista de $(s,a)$.[RLBook2018; FairRLClass2024 — evidencia: fuerte][^1][^2] En estas condiciones, maximizar el retorno esperado es equivalente a minimizar una pérdida de clasificación dependiente del coste, como se ha demostrado en trabajos de clasificación sensible al coste y de bandits contextuales.[RLCC2023; ContextBanditsCost2023 — evidencia: fuerte][^4][^6]

3. **Reward design**

> En lugar de utilizar una recompensa 0/1, definimos una función de recompensa sensible al coste que penaliza de forma más severa las falsas negativas (bloquear ataques que se etiquetan como benignos) que las falsas positivas, siguiendo la literatura de clasificadores sensibles al coste y de IDS.[RLCC2023; CostSensitiveIDS; FP/FN-CostModel — evidencia: moderada‑fuerte] Esta recompensa se integra directamente en el objetivo de QRDQN, de forma que la política aprendida aproxima explícitamente una política de defensa basada en costes, en lugar de una mera tasa de aciertos.[^15][^14][^4]

4. **Limitaciones y honestidad**

> No obstante, esta formulación sigue siendo una aproximación controlada: el entorno no modela explícitamente la dinámica de la red, los flujos se muestrean de un dataset estático y las acciones del agente no influyen en estados futuros, por lo que la estructura secuencial de un problema RL completo está ausente.[RLBook2018; RLDiag2023 — evidencia: fuerte/moderada] En consecuencia, el problema es matemáticamente cercano a la clasificación supervisada sensible al coste, y cualquier ventaja de RL debe demostrarse empíricamente frente a baselines supervisados robustos (por ejemplo, Random Forest) bajo protocolos de evaluación estrictos en CICIDS2017.[^13][^1]

5. **Puente hacia defensa secuencial**

> A pesar de estas limitaciones, la formulación de la clasificación de flujos como entorno de RL proporciona un marco coherente para integrar, en trabajos futuros, decisiones realmente secuenciales como bloquear conexiones, generar alerts o adaptar umbrales en tiempo real, en línea con trabajos que utilizan DQN para mejorar la detección en escenarios de IDS abiertos o en sistemas ciberfísicos.[DQNIDS2026; DCIDS2024; DRL-NIDS-Survey — evidencia: moderada] En esta tesis, se explora esta formulación en un escenario estático y offline (Fase 1) y se valida externamente sobre tráfico de laboratorio (Fase 2), sin afirmar que RL sea intrínsecamente superior a la clasificación supervisada, sino evaluando empíricamente sus ventajas y desventajas bajo un marco experimental riguroso.[^18][^11][^10]

***

## 7. Codex handoff

Guidelines for using this dossier in:

### State of the art

- In the **RL for NIDS** subsection, briefly mention works that treat intrusion detection as RL classification (e.g., DQLIDS2021 on NSL‑KDD, DQNIDS2026 and DCIDS2024 for open‑set NIDS) and cost‑sensitive RL classifiers (RLCC2023).[^11][^5][^10][^4]
- Emphasize that these approaches generally:
    - Use sample features as state and labels as actions.
    - Define rewards based on correct/incorrect classification or cost matrices.
    - Operate on static datasets, sometimes with open‑set or cost‑sensitive twists.
- Mark evidence strength and caveats for each (e.g., NSL‑KDD is outdated; open‑set NIDS uses RL mainly as a decision layer).


### RL environment design

- In the **“Diseño del entorno de RL”** section, follow the subsection template above.
- Clearly define:
    - State $s$ as the preprocessed flow feature vector.
    - Action $a\in\{0,1\}$ as PERMIT/BLOCK.
    - Reward function $r(s,a)$ with explicit numerical values reflecting FP/FN cost asymmetry (to be filled from your design).
    - Episode structure (e.g., fixed‑length episodes over minibatches).
- Reference RLBook2018 for formal MDP/bandit background and RLCC2023 + ZeroDayRL‑Protocol for the specific classification‑as‑RL pattern.[^3][^4][^1]


### Methodological discussion

- Add a short **“Discusión metodológica”** subsection that:
    - States explicitly that your environment is closer to a contextual bandit than to a fully sequential MDP (no action‑dependent transitions).
    - Explains that RL is used here as a unified framework for cost‑sensitive policy learning and as a stepping stone towards future adaptive defence, not as a claim that RL is inherently superior for static classification.
    - Commits to comparing QRDQN against strong supervised baselines (Random Forest, maybe a simple MLP) on identical splits and features.
- Use cautious language and the **arguments against** section to avoid overclaiming: when in doubt, phrase RL benefits as *“posibles ventajas”* that are **empirically evaluated**, not assumed.

If Codex adheres to these instructions, the resulting thesis text will present your classification‑as‑RL formulation as a **conscious modelling choice with clear benefits and clear limits**, grounded in existing literature and carefully contrasted with conventional supervised learning.
<span style="display:none">[^20][^21][^22][^23][^24][^25][^26][^27][^28][^29][^30][^31][^32][^33][^34][^35][^36][^37][^38][^39][^40][^41][^42][^43][^44][^45][^46][^47][^48][^49]</span>

<div align="center">⁂</div>

[^1]: https://apxml.com/courses/intro-to-reinforcement-learning/chapter-1-foundations-reinforcement-learning/states-actions-rewards

[^2]: https://arxiv.org/html/2407.10629v1

[^3]: https://bio-protocol.org/exchange/minidetail?id=20788741\&type=30

[^4]: https://link.springer.com/10.1007/s11432-021-3775-4

[^5]: https://deep.ai/publication/deep-q-learning-based-reinforcement-learning-approach-for-network-intrusion-detection

[^6]: https://openreview.net/attachment?id=ak5A0IrCJ5\&name=pdf

[^7]: https://en.wikipedia.org/wiki/Reinforcement_learning

[^8]: https://www.semanticscholar.org/paper/Reinforcement-learning-based-cost-sensitive-for-Zhang-Fan/238e25c5cb8f81cd3d1cb5cba8e6a78a1a8f590d

[^9]: https://openreview.net/pdf?id=ak5A0IrCJ5

[^10]: https://eprints.ncl.ac.uk/295350

[^11]: https://www.ndss-symposium.org/ndss-paper/auto-draft-729/

[^12]: https://www.ndss-symposium.org/wp-content/uploads/sdiotsec26-97.pdf

[^13]: https://arxiv.org/abs/2302.10261

[^14]: https://www.scirp.org/journal/paperinformation?paperid=43038

[^15]: https://www.sciencedirect.com/science/article/abs/pii/S0167404821003230

[^16]: https://arxiv.org/abs/1710.10044

[^17]: https://www.semanticscholar.org/paper/2cdbddb14304434aef9fdb3d22e04fb89a742330

[^18]: https://arxiv.org/html/2410.07612v2

[^19]: https://research.sabanciuniv.edu/id/eprint/48877/1/10569767.pdf

[^20]: http://arxiv.org/pdf/2501.07502.pdf

[^21]: https://arxiv.org/pdf/2302.00270.pdf

[^22]: https://arxiv.org/pdf/1806.01946.pdf

[^23]: http://arxiv.org/pdf/2312.16566.pdf

[^24]: https://jmir.org/api/download?alt_name=jmir_v22i7e18477_app1.pdf\&filename=3594d1f00d24cd81f107de19f78ea67a.pdf

[^25]: https://linkinghub.elsevier.com/retrieve/pii/S0004370224000821

[^26]: http://arxiv.org/pdf/1706.03741.pdf

[^27]: https://arxiv.org/html/2403.10946v2

[^28]: https://digitalcommons.calpoly.edu/cgi/viewcontent.cgi?params=%2Fcontext%2Ftheses%2Farticle%2F4265%2F\&path_info=Asmita_s_Thesis.pdf

[^29]: https://arxiv.org/abs/2003.03051

[^30]: https://ijcsm.researchcommons.org/ijcsm/vol6/iss2/9/

[^31]: https://www.neuralnet.ai/designing-your-own-open-ai-gym-compatible-reinforcement-learning-environment/

[^32]: https://www.facebook.com/groups/36085277798/posts/10158426317287799/

[^33]: https://ieeexplore.ieee.org/iel8/6287639/6514899/11247829.pdf

[^34]: http://scis.scichina.com/en/2023/212201.pdf

[^35]: https://github.com/openai/gym

[^36]: https://www.semanticscholar.org/paper/296094909b3a3524b8265410b6f9c4c63ebc9de8

[^37]: https://www.nature.com/articles/s41598-022-19443-7

[^38]: https://link.springer.com/10.1007/s12008-020-00715-3

[^39]: https://ieeexplore.ieee.org/document/10575541/

[^40]: https://ieeexplore.ieee.org/document/10223707/

[^41]: https://ieeexplore.ieee.org/document/11336652/

[^42]: https://ieeexplore.ieee.org/document/11176031/

[^43]: https://aclanthology.org/2026.eacl-long.208.pdf

[^44]: https://www.sciencedirect.com/science/article/abs/pii/S2214212623001928

[^45]: http://scis.scichina.com/en/2022/182102.pdf

[^46]: https://arxiv.org/pdf/2603.03752.pdf

[^47]: https://slogix.in/machine-learning/research-topics-in-deep-reinforcement-learning-for-classification/

[^48]: http://www.arxiv.org/pdf/2409.13007.pdf

[^49]: https://www.linkedin.com/posts/jonathansilvasantos_finegrainedrecognition-reinforcementlearning-activity-7434821755883429888-nb1_

