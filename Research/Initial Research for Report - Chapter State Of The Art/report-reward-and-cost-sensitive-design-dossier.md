<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Reward and Cost-Sensitive Design Dossier

## 1. Conceptual foundation

**Claim 1.** In NIDS the costs of false negatives (FNs) and false positives (FPs) are structurally different: a false negative (malicious traffic classified as benign) can lead to compromise, data loss, or service interruption, while a false positive mainly causes operational overhead and potential loss of availability. Evidence strength: strong. Caveat: for some environments (e.g., safety‑critical industrial control), certain FPs can also be highly disruptive; costs must be contextualized. Use: justify asymmetric cost modeling and reward design.[^1][^2][^3]

Formal cost models for IDS decompose **total expected cost** into components such as damage cost from successful intrusions, cost of manual and automated responses, and operating costs, and explicitly treat FP and FN reaction costs as different terms in this sum. Evidence strength: strong. Caveat: parameters are site‑specific and often estimated rather than measured. Use: motivate mapping business/security costs into reward weights.[^2][^4][^3]

Risk‑oriented work shows that these asymmetries should be reflected in decision policies: optimizing only global accuracy can produce an operating point with unacceptably high FN risk even if the overall error rate is low. Evidence strength: moderate. Caveat: many models are stylized (e.g., financial transactions, simulated cost curves). Use: motivate using cost‑sensitive metrics and reward functions rather than plain accuracy.[^5][^1]

***

## 2. Source matrix

Key sources on reward/cost design and FP/FN trade‑offs.


| Citation key | Source | Year | Domain | Reward / cost design | Metrics used | Main idea | Limitation | Relevance to my thesis |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| Baayer2014-FPOpt | Baayer et al., “False Positive Responses Optimization for Intrusion Detection System,” J. Info. Security | 2014 | IDS cost model | Analytical cost model with separate terms for FP and FN “missed reactions”; search for operating interval that minimizes total cost | Expected cost vs FP rate, damage cost | Shows FP and FN reaction costs can be combined into an optimal operating region (not a single accuracy metric) | Uses simplified simulations and generic cost parameters | Theoretical basis for arguing that IDS operating point should be chosen via FP/FN cost trade‑offs, not raw accuracy[^3][^5] |
| Lee2002-CostSensitiveIDS | Lee et al., “Toward Cost-Sensitive Modeling for Intrusion Detection and Response,” ACM CCS workshop | 2002 | IDS modeling | Defines cost model including development, operational, damage, and response costs; proposes cost-sensitive ML to minimize expected total cost | Detection rate, FP rate, total expected cost | Classic formulation of cost-sensitive IDS; provides methodology to derive cost matrices from business assumptions | Uses older datasets and pre‑DL ML; cost figures are site‑specific | Direct support for using cost-sensitive objectives and aligning RL rewards to site-specific costs[^2][^4] |
| Donchev2021-FP_FN_Risk | Donchev et al., “Impact of false positives and false negatives on security risks in transactions under threat,” TRUSTBUS | 2021 | Transaction security (POMDP) | Models transaction checking as POMDP; analyzes how varying FP/FN parameters affects overall risk | Risk metrics over time, optimal strategy parameters | Demonstrates that FP and FN have different contributions to risk; shows how optimal policy changes with their relative costs | Focused on transactions, not NIDS; but generic conclusions | Gives a risk-theoretic argument that FP/FN weights in your reward should reflect actual risk preferences[^1] |
| CSE-IDS2022 | Gupta et al., “CSE-IDS: Using cost-sensitive deep learning and ensemble algorithms to handle class imbalance in NIDS,” Computers \& Security | 2022 | Supervised NIDS | Introduces cost-sensitive deep nets and ensembles that weight misclassification costs per class, especially rare attacks | Accuracy, precision, recall, F1 per class | Shows cost-sensitive training improves detection of minority attacks at acceptable FP cost | Purely supervised; no RL; on public datasets with limited external validation | Direct evidence that cost-sensitive losses improve NIDS behavior; analogous to cost-sensitive rewards in RL[^6] |
| RLMetaCost2018 | Marín et al., “Applying Cost-Sensitive Classifiers with Reinforcement Learning to IDS,” in *Advances in Intelligent Systems and Computing* | 2018 | ML+RL for IDS | Uses DQN as meta-classifier to adjust decisions of base MLP using reward strategies that prioritize certain error types; different reward schemes implement different FP/FN trade‑offs | FP, FN counts, cost-weighted errors | Demonstrates that by changing reward design, RL can bias an IDS towards reducing a specific error type (up to 100% reduction for targeted error) without runtime cost | Evaluated on two older datasets; extra training complexity vs cost-sensitive supervised methods | Direct RL precedent: reward design used explicitly to control FP vs FN emphasis; supports your use of reward variants[^7] |
| MARL-IDS2024 | “Multi-agent Reinforcement Learning-based Network Intrusion Detection System,” arXiv | 2024 | MARL NIDS | Multi-agent DQN; L1 agents per attack type; reward gives positive for correct classifications, negative for misclassifications, with class-dependent weighting for minority/critical attacks | Accuracy, F1 per class, FPR | Uses class-weighted rewards and cost-sensitive loss to improve detection of minority attack types; explicitly discusses reward design for imbalance | Very high reported accuracy and extremely low FPR; needs careful interpretation | Example of class-weighted reward in RL-IDS; supports weighted penalties for specific attack types or FN-heavy design[^8][^9] |
| APT-RL-Reward2025 | “Evaluating Reinforcement Learning Reward Functions for APT Detection in IIoT Systems,” IEEE | 2025 | IIoT APT detection (RL IDS) | Compares three reward functions: (i) weighted precision–recall, (ii) cost-sensitive (FN‑heavy), (iii) balanced accuracy; measures impact on detection trade‑offs | Precision, recall, F1, FPR, cost-based metrics | Empirically shows that cost-sensitive reward better reduces FNs in high-stakes IIoT, whereas balanced accuracy reward yields more symmetric trade‑off | Specific to CICAPT-IIoT2024 dataset; depends on particular threat model | Direct evidence that reward design significantly shifts FP/FN behavior in RL-IDS; supports running controlled reward variants[^10] |
| DynReward-IoT-IDS2024 | “A Dynamic Reward-Based Deep Reinforcement Learning for IoT Intrusion Detection,” IEEE | 2024 | IoT DRL-IDS | Designs dynamic reward function sensitive to multi-class samples and detection difficulty to improve recognition of various attack vectors | Accuracy, per-class precision/recall | Shows dynamic, class-aware rewards can improve multi-class performance on Bot-IoT | Risk of reward over-tuning and complex behavior; no external validation | Example of adaptive reward shaping; useful to cite when discussing shaping risks and design choices[^11] |
| UnsupervisedReward-IDS2024 | El Ouadrhiri et al., “Towards an Unsupervised Reward Function for a Deep Reinforcement Learning Based IDS,” HAL | 2024 | DRL-IDS (unsupervised) | Proposes unsupervised reward using a normality score (autoencoder/anomaly detector) instead of labels to support continual learning | Detection rate, FPR, anomaly scores | Highlights limitations of label-based reward (no continual learning) and explores unsupervised reward design | Methodologically complex; depends on quality of anomaly model; not directly cost-sensitive | Shows alternative reward design paradigms and warns that label-only reward may be insufficient for long-term adaptation[^12][^13] |
| RiskAware-DRL-IDS2026 | “A Risk-Aware Deep Reinforcement Learning Framework for AI-Driven Intrusion Detection,” IJNCSA | 2026 | Automotive IDS | Combines CNN–BiLSTM detection with risk model (attack probability, ECU criticality, vehicle speed) and DQN mitigation policy minimizing system damage | Detection rate, system damage cost, latency | Integrates detection with severity modeling; DQN’s reward directly encodes residual damage cost | Focused on CAN bus/automotive; requires detailed risk model | Strong precedent for aligning reward with physical/system-level damage cost instead of only classification accuracy[^14] |
| RL-IDS-EdgeCloud2026 | “Adaptive Intrusion Detection in Edge-Cloud Environments Using RL,” IEEE | 2026 | Edge-cloud RL-IDS | RL-IDS reward designed to reduce FNs and adapt to changing attack behavior; uses CIC-IDS2017 flows and balances detection vs false alarms | Accuracy, precision, recall, F1, AUC, FPR | Shows RL reward can be tuned to prioritize FN reduction in dynamic environments, outperforming CNN/LSTM/XGBoost baselines | Very high reported metrics; reward specifics may be under-specified | Example of FN‑heavy reward in practical RL-IDS using CIC-IDS2017; supports your CICIDS2017-based design[^15] |


***

## 3. Reward patterns in the literature

### 3.1 Simple correct/incorrect rewards

Many RL‑IDS and DRL‑security works use a **binary reward**: positive for correct classification, negative or zero for incorrect, often with the same magnitude across classes. Evidence strength: moderate (common but often under‑documented). Caveat: this ignores class imbalance and risk asymmetry, effectively optimizing accuracy. Use: contrast with your more nuanced design.[^16][^17][^18]

### 3.2 Class-weighted rewards

Multi‑agent RL‑IDS designs and MARL frameworks explicitly weight rewards by class or attack criticality: correct detection of minority/critical attacks yields higher reward; misclassifying them yields stronger penalties. Evidence strength: strong for existence; effectiveness shown on CICIDS2017. Caveat: weight choices are often heuristic and tuned for metrics, not derived from explicit cost models. Use: precedent for class- and risk-aware reward in RL‑IDS.[^9][^8]

### 3.3 False-negative-heavy (cost-sensitive) rewards

APT‑detection RL and adaptive RL‑IDS for edge‑cloud environments use cost‑sensitive reward functions that penalize FNs more heavily than FPs, reflecting that missing an APT or critical attack is more damaging than triggering an alert. Evidence strength: strong. Caveat: papers sometimes tune weights empirically; need to avoid “reward hacking” where weights are adjusted until metrics look good. Use: direct support for FN‑heavy reward design in your thesis.[^10][^15][^14]

### 3.4 Precision- / recall-oriented rewards

Some RL‑IDS works define reward proportional to **precision**, **recall**, or F‑scores over recent windows, effectively optimizing batch‑level metrics rather than per‑sample correctness. Evidence strength: moderate. Caveat: delayed and non‑local reward complicates credit assignment; risk of instability. Use: mention as advanced design but probably avoid for your first implementation.[^10]

### 3.5 Dynamic/adaptive rewards

Dynamic reward functions adjust penalties depending on class difficulty or misclassification patterns over time, often to boost minority or under‑detected classes in IoT DRL‑IDS. Evidence strength: moderate. Caveat: increases complexity and risk of overfitting reward to dataset quirks; can be hard to interpret. Use: cite under “reward shaping risks”.[^11]

### 3.6 Risk-based/system-cost rewards

Risk‑aware DRL frameworks translate detection outcomes into system‑level damage costs (e.g., combining attack probability with physical criticality and speed) and train DQN to minimize residual damage. Evidence strength: moderate–strong in specific domains. Caveat: requires a domain‑specific risk model; not trivial for generic NIDS. Use: conceptual justification that rewards should, in principle, approximate business/security costs.[^14]

### 3.7 Unsupervised/normality-score rewards

Unsupervised reward designs use anomaly/normality scores (from autoencoders, density estimators) as reward feedback to allow continual learning even without labels. Evidence strength: moderate (early; more complex). Caveat: not directly cost-sensitive; depends on anomaly model calibration. Use: demonstrate broader design space; motivates clean, label‑based design in your thesis as a controlled first step.[^12]

***

## 4. Recommended reward design for my thesis

Let $y \in \{0,1\}$ be the true label (0 = benign, 1 = attack) and $a \in \{0,1\}$ the agent’s action (0 = PERMIT, 1 = BLOCK). Let $C_{\text{FN}}>0$ and $C_{\text{FP}}>0$ represent relative costs inferred from literature and domain assumptions.

### 4.1 Variant A – Main balanced cost-sensitive reward

**Formula (per step)**

$$
r = \begin{cases}
+1 & \text{if } a = y \\
-C_{\text{FN}} & \text{if } y=1, a=0 \quad (\text{FN}) \\
-C_{\text{FP}} & \text{if } y=0, a=1 \quad (\text{FP})
\end{cases}
$$

with e.g. $C_{\text{FN}} = 5, C_{\text{FP}} = 1$.

**Rationale.** Encodes that FNs are more costly than FPs while still rewarding all correct decisions equally; consistent with cost models and NIDS cost-sensitive literature. Evidence strength: strong for direction (FN>FP), not for exact numeric ratio. Use: main reward in thesis.[^6][^3][^2]

**Expected behavior.** Agent biases towards blocking suspicious flows (reduce FNs) but still experiences penalty for FPs, resulting in a moderate “defensive” policy.

**Risk.** If $C_{\text{FN}}$ is set arbitrarily or tuned on test data, the policy may be over‑tuned to dataset idiosyncrasies; must justify weights qualitatively and ideally with small sensitivity analysis.[^7][^10]

**Metrics to check.**

- Per‑class recall for attacks (FNR).
- FPR on benign flows.
- Cost-weighted metric such as estimated average cost per sample using the same $C_{\text{FN}},C_{\text{FP}}$.
- Macro‑F1.


### 4.2 Variant B – Strict FN‑penalty reward

**Formula**

Same structure, but with stronger asymmetry, e.g. $C_{\text{FN}} = 10, C_{\text{FP}} = 1$.

**Rationale.** Models “high‑stakes” environments where missing an attack is extremely costly (APTs, critical infrastructure); mirrors cost‑sensitive RL‑IDS for IIoT APTs and risk‑aware DRL in automotive. Evidence strength: moderate‑strong.[^14][^10]

**Expected behavior.** Very aggressive blocking; significantly lower FNs, higher FPs. Good for exploring the Pareto frontier of detection vs operational cost.

**Risk.** May lead to pathological behavior (blocking almost everything) if the classifier is weak or the dataset distribution is skewed; policy may not be acceptable in practice.

**Metrics to check.**

- FNR on critical attack classes (should be very low).
- FPR and proportion of benign traffic blocked (operational feasibility).
- Cost‑per‑sample (using same cost model) to check if extreme FN reduction actually reduces expected cost.


### 4.3 Variant C – Availability-conscious reward (FP‑penalty heavy)

**Formula**

$$
r = \begin{cases}
+1 & a = y\\
-C_{\text{FN}} & y=1,a=0\\
-C_{\text{FP}} & y=0,a=1
\end{cases}
$$

with $C_{\text{FN}} = 3, C_{\text{FP}} = 5$.

**Rationale.** Models environments where service availability and user experience are critical (e.g., customer‑facing services), and frequent blocking is very costly, while some low‑severity attacks may be tolerable or mitigated by other layers (e.g., rate limiting, WAF). Evidence strength: moderate (conceptual, less NIDS‑specific).[^1][^2]

**Expected behavior.** Agent becomes conservative in blocking, aiming for low FPR even at the cost of higher FNR; useful as a stress‑test scenario and demonstration of how reward aligns with business priorities.

**Risk.** Can significantly increase FNs, making it unsuitable as primary policy in high‑risk settings; must be clearly marked as exploratory.

**Metrics to check.**

- FPR (should be low).
- FNR and missed critical attacks.
- Cost‑per‑sample under availability‑heavy cost model.

***

## 5. Experimental protocol for reward variants

Goals: compare reward designs **without exploding the experimental matrix** and **without implicitly tuning on test data**.

1. **Fixed dataset splits.**
    - Use a **single, fixed train/validation/test split** on CICIDS2017 (and the lab traffic if available), defined in your methodology chapter.
    - Never alter splits between reward variants.[^19][^20]
2. **Reward variants treated as *design choices*, not tuned on test.**
    - Pre‑define 2–3 reward variants (A/B/C) with **a priori** cost ratios based on literature and domain judgment, not on performance.
    - If you adjust costs, do so using **only the validation set** and document all changes; do not re‑use the validation set as test.
3. **Same training budget.**
    - For each reward variant and algorithm (QRDQN, possibly DQN), use the **same number of environment steps, gradient updates, and hyperparameters**, changing only the reward function.
    - Fix random seeds or run **K seeds** (e.g., 5) for each combination and report mean ± std.
4. **Common metrics.**
    - On the held‑out test set, compute:
        - Accuracy;
        - Precision, recall, F1 for benign and attack (macro‑F1);
        - FPR and FNR explicitly;
        - Cost‑per‑sample estimate using each reward’s $C_{\text{FN}},C_{\text{FP}}$ (same formula as reward but evaluated on test predictions).[^3][^2]
5. **Small matrix.**
    - Example:
        - Algorithms: RF baseline, QRDQN.
        - Rewards: A (main), B (FN‑strict). C can be added if you have time.
        - Seeds: 5.
    - Total RL runs: 2 rewards × 5 seeds = 10 per algorithm per dataset, which is manageable.
6. **Analysis.**
    - Compare **policy behavior** across reward variants: show how shifting $C_{\text{FN}}/C_{\text{FP}}$ changes FNR/FPR and cost‑per‑sample.
    - Avoid selecting reward variant post‑hoc based solely on test performance; instead, discuss trade‑offs qualitatively (e.g., “Variant B reduces FNR by X but increases FPR by Y; depending on deployment priorities, this may or may not be acceptable”).

***

## 6. Thesis-ready text

### 6.1 Subsection: “Asimetría entre falsos positivos y falsos negativos”

> En un sistema de detección de intrusiones, los costes asociados a los falsos positivos (FP) y a los falsos negativos (FN) no son equivalentes. Un FN —tráfico malicioso clasificado como benigno— puede conducir a compromisos, pérdida de datos o interrupciones de servicio, con un impacto potencialmente elevado en la organización.[Baayer2014-FPOpt; Lee2002-CostSensitiveIDS] Por el contrario, un FP —tráfico benigno bloqueado o marcado como ataque— genera sobre todo costes operativos (análisis manual, degradación de disponibilidad, molestias al usuario), que suelen ser significativos pero menos críticos que el éxito de un ataque.[^2][^3]
> Modelos clásicos de coste para IDS descomponen el coste total esperado en componentes de desarrollo, operación, daño por intrusión y respuesta, e incluyen términos diferenciados para FP y FN, mostrando que el sistema debe operar en una región donde el incremento de FP compensa razonablemente la reducción de FNs.[Baayer2014-FPOpt; Donchev2021-FP_FN_Risk] Estos trabajos motivan el uso de métricas y funciones objetivo sensibles al coste, en lugar de optimizar únicamente la exactitud global, ya que distintas configuraciones de FP/FN pueden tener riesgos muy distintos aunque compartan la misma tasa de acierto.[^3][^1]

### 6.2 Subsection: “Diseño de la recompensa sensible al coste”

> En el entorno de aprendizaje por refuerzo, definimos la recompensa por paso como una aproximación explícita de la función de coste del IDS. Cada flujo se representa por su etiqueta real $y\in\{0,1\}$ (benigno/ataque) y la acción del agente $a\in\{0,1\}$ (PERMIT/BLOCK). La recompensa se define como $r=+1$ si la decisión es correcta y como un valor negativo proporcional al coste relativo del error cuando la decisión es incorrecta, es decir, penalizando más severamente las falsas negativas (ataques permitidos) que las falsas positivas (tráfico benigno bloqueado).[Lee2002-CostSensitiveIDS; CSE-IDS2022][^6][^2]
> Esta recompensa tiene la forma $r=-C_{\text{FN}}$ cuando $y=1$ y $a=0$, y $r=-C_{\text{FP}}$ cuando $y=0$ y $a=1$, con $C_{\text{FN}}>C_{\text{FP}}>0$. La elección concreta de $C_{\text{FN}}$ y $C_{\text{FP}}$ se guía por la literatura de modelado de costes en IDS y por consideraciones de negocio (por ejemplo, mayor severidad para compromisos de confidencialidad o disponibilidad), en lugar de ser ajustada directamente sobre el conjunto de prueba.[Baayer2014-FPOpt; Donchev2021-FP_FN_Risk] Trabajos previos han demostrado que recompensas sensibles al coste y estrategias de refuerzo permiten desplazar el comportamiento de un IDS hacia la reducción de un tipo de error concreto, e incluso reducir prácticamente a cero una clase de error a costa de aumentar la otra, lo que refuerza la importancia de documentar y justificar explícitamente la función de recompensa utilizada.[Marín2018-RLMetaCost; APT-RL-Reward2025][^7][^10][^1][^3]

### 6.3 Paragraph on limitations

> Sin embargo, el diseño de recompensas en sistemas de seguridad presenta riesgos metodológicos. Recompensas excesivamente complejas o adaptativas pueden favorecer comportamientos difíciles de interpretar y sobreajustar el agente a peculiaridades del conjunto de datos, especialmente si los pesos de la recompensa se ajustan iterativamente hasta obtener métricas “atractivas” en el conjunto de prueba.[DynReward-IoT-IDS2024; UnsupervisedReward-IDS2024] Además, incluso una recompensa sensible al coste sigue siendo una aproximación de la realidad: los costes reales de FP y FN dependen del contexto organizativo y pueden cambiar con el tiempo. Por ello, en esta tesis se exploran únicamente un pequeño número de variantes de recompensa predefinidas, se evalúan bajo el mismo protocolo experimental y se comparan con baselines supervisados robustos, sin asumir que una determinada elección de recompensa sea universalmente óptima.[^11][^12]

***

## 7. Codex handoff

How Codex should use this material:

### RL environment

- In the **environment description**, Codex should:
    - Define the per‑step reward function explicitly with equations for at least one main variant (Balanced A) and mention the others (FN‑heavy B, availability‑conscious C) as alternative experiments.
    - Clarify that $C_{\text{FN}}$ and $C_{\text{FP}}$ are derived from **risk and cost considerations**, referencing Baayer2014-FPOpt and Lee2002-CostSensitiveIDS, not tuned on the test set.[^2][^3]
    - Explain that the RL objective (expected discounted return) is thus an approximation of minimizing expected IDS cost, consistent with cost-sensitive NIDS literature like CSE-IDS and RLMetaCost.[^6][^7]


### Experimental design

- In the **experimental design** chapter, Codex should:
    - Describe the controlled reward‑variant experiment: same splits, same hyperparameters, only changing reward weights, with multiple seeds per configuration.
    - List the evaluation metrics for each policy: accuracy, precision, recall, macro‑F1, FPR, FNR, and estimated cost‑per‑sample using the same $C_{\text{FN}},C_{\text{FP}}$ as in the reward.[^10][^3]
    - Explicitly state that **no hyperparameters or reward weights** are tuned using the test set; if any tuning is needed, it is done on a separate validation set.


### Results discussion

- In the **results** chapter, Codex should:
    - Present tables comparing RF vs QRDQN (and optionally DQN) under the main reward design, plus a smaller table showing how changing $C_{\text{FN}}/C_{\text{FP}}$ (Variants A vs B) shifts the FNR/FPR trade‑off.
    - Interpret results in terms of **cost trade‑offs**: e.g., “Variant B achieved a 40% reduction in FNR at the cost of a 25% increase in FPR; depending on deployment priorities, this may be acceptable or not.”
    - Avoid claiming that a particular reward is “optimal”; instead, discuss how it approximates a given risk profile and where it might fail (e.g., too many FPs in availability‑sensitive contexts).[^14][^10]

If Codex follows these guidelines, the thesis will present reward and cost-sensitive design as **deliberate, theoretically grounded choices** aligned with existing IDS cost models and RL‑IDS works, and will clearly separate design assumptions from empirical findings.
<span style="display:none">[^21][^22][^23][^24][^25]</span>

<div align="center">⁂</div>

[^1]: https://repository.londonmet.ac.uk/6776/

[^2]: https://researchconnect.suny.edu/en/publications/toward-cost-sensitive-modeling-for-intrusion-detection-and-respon/

[^3]: https://www.scirp.org/journal/paperinformation?paperid=43038

[^4]: https://www.fsl.cs.stonybrook.edu/docs/cost-acm_ccs/index.html

[^5]: https://file.scirp.org/Html/1-7800184_43038.htm

[^6]: https://www.scribd.com/document/752672314/1-s2-0-S0167404821003230-main

[^7]: https://www.ccs.upm.es/research/publications/applying-cost-sensitive-classifiers-with-reinforcement-learning-to-ids/

[^8]: https://arxiv.org/html/2407.05766v1

[^9]: https://www.themoonlight.io/fr/review/multi-agent-reinforcement-learning-based-network-intrusion-detection-system

[^10]: https://ieeexplore.ieee.org/document/11013675/

[^11]: https://ieeexplore.ieee.org/document/10865958/

[^12]: https://hal.science/hal-04818190v1/document

[^13]: https://ieeexplore.ieee.org/document/10851732/

[^14]: https://ijcnc.com/2026/04/17/ijnsa-07-2/

[^15]: https://ieeexplore.ieee.org/document/11479986/

[^16]: https://ieeexplore.ieee.org/document/11165835/

[^17]: https://ieeexplore.ieee.org/document/11484628/

[^18]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12589504/

[^19]: http://www.scitepress.org/DigitalLibrary/Link.aspx?doi=10.5220/0006105602530262

[^20]: http://link.springer.com/10.1007/978-3-030-25109-3_9

[^21]: https://ieeexplore.ieee.org/document/11076108/

[^22]: https://link.springer.com/10.1007/s10586-025-05589-2

[^23]: https://securitybulldog.com/blog/reinforcement-learning-for-intrusion-detection-overview/

[^24]: https://www.themoonlight.io/de/review/multi-agent-reinforcement-learning-based-network-intrusion-detection-system

[^25]: https://ieeexplore.ieee.org/document/10911733

