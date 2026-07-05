<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# QRDQN / Distributional RL Research Dossier

## 1. DQN foundation

**Claim 1 (standard DQN setup).** In Deep Q‑Networks (DQN), an agent interacts with an environment in discrete time steps, observing a state $s_t$, choosing an action $a_t$, receiving a scalar reward $r_t$, and moving to a next state $s_{t+1}$.[Mnih2015-DQN — evidence: strong; caveat: originally Atari with image states; use: baseline RL background][^1]
The goal is to learn the **action‑value function** (Q‑function)

$$
Q^\*(s,a) = \mathbb{E} \big[ \sum_{k=0}^{\infty} \gamma^k r_{t+k} \mid s_t=s, a_t=a \big],
$$

which satisfies the Bellman optimality equation

$$
Q^\*(s,a) = \mathbb{E}\big[ r_t + \gamma \max_{a'} Q^\*(s_{t+1}, a') \mid s_t=s, a_t=a \big].
$$[^1]

DQN approximates $Q_\theta(s,a)$ with a neural network and minimizes the squared **temporal‑difference (TD) error** between predicted Q and a target:

$$
y_t = r_t + \gamma \max_{a'} Q_{\theta^-}(s_{t+1}, a'), \quad
\mathcal{L}(\theta) = \mathbb{E}[(y_t - Q_\theta(s_t,a_t))^2].
$$[^1]

Key engineering components:

- **Target network** $Q_{\theta^-}$: a periodically updated copy of the Q‑network to stabilize training by providing a fixed target for several updates.[^1]
- **Replay buffer**: stores $(s_t,a_t,r_t,s_{t+1},\text{done}_t)$ tuples and samples random minibatches to break correlations and reuse past experience.[^1]
- **$\epsilon$‑greedy exploration**: during training, choose a random action with probability $\epsilon$, otherwise $a_t=\arg\max_a Q_\theta(s_t,a)$; $\epsilon$ is usually annealed from a high initial value to a small final value.[^1]

In your thesis, the DQN “foundation” is the baseline algorithm that QRDQN extends: same replay buffer, target network, and $\epsilon$-greedy policy, but with a different representation of the value function.

***

## 2. From DQN to Distributional RL

**Claim 2 (limitation of mean-value DQN).** Standard DQN only estimates the **expected return** $Q(s,a) = \mathbb{E}[Z(s,a)]$, which collapses all randomness (stochastic rewards, transitions, exploration) into a single scalar; this may hide important information about risk and variability of outcomes.[Bellemare2017-Distributional; Dabney2017-QRDQN — evidence: strong][^2][^3]
Caveat: Many control tasks still work well with just the mean; the benefit of modeling the full distribution is empirical and task‑dependent.

**Distributional RL** instead models the **return distribution** $Z(s,a)$ itself, where

$$
Z(s,a) \overset{d}{=} \sum_{k=0}^\infty \gamma^k r_{t+k}
$$

is treated as a random variable.[^3]
Bellemare et al. define a **distributional Bellman operator** that pushes forward distributions:

$$
\mathcal{T} Z(s,a) \overset{d}{=} r(s,a) + \gamma Z(s',a^*), \quad a^* = \arg\max_{a'} \mathbb{E}[Z(s',a')].
$$[^3]

**Claim 3 (why distributional estimates help).** Experiments on Atari show that approximating the full return distribution and training with a distributional Bellman update can improve learning speed and final performance, and also provides richer uncertainty information that can be exploited for exploration and risk‑sensitive behavior.[Bellemare2017-Distributional; Dabney2017-QRDQN; Dabney2018-IQN — evidence: strong][^4][^2][^3]
Caveat: Most evidence is from Atari and similar benchmarks; benefits in tabular or low‑noise tasks may be smaller.

In your NIDS setting, the stochasticity comes from random sampling of flows and potentially from reward noise or label noise; distributional RL can, in principle, provide:

- Better gradients via richer TD targets.
- A way to assess risk (e.g., low‑quantile returns for risky actions like PERMIT).

***

## 3. QRDQN explanation

### 3.1 Quantiles and return distribution

**Claim 4 (quantile parameterization).** QRDQN represents the return distribution $Z(s,a)$ as a set of $N$ quantiles $\theta_i(s,a)$, approximating the inverse CDF at fixed fractions $\tau_i\in(0,1)$ (e.g., $\tau_i = (i+0.5)/N$).[Dabney2017-QRDQN — evidence: strong][^2]
Intuitively, instead of a single expected return, the network outputs multiple “levels” (e.g., 0.1, 0.5, 0.9 quantiles) that summarize the distribution.

For each state–action pair, QRDQN approximates a **staircase distribution** made of $N$ equally weighted atoms at locations $\theta_i(s,a)$.

### 3.2 Quantile regression and quantile Huber loss

The learning objective is to make the predicted quantiles $\theta_i(s_t,a_t)$ match a target distribution built from TD‑updated quantiles at the next state.

Given a sampled transition $(s_t,a_t,r_t,s_{t+1})$, QRDQN:

1. Computes target quantile locations

$$
y_j = r_t + \gamma \,\theta_j^{-}(s_{t+1}, a^*), \quad a^* = \arg\max_{a'} \frac{1}{N}\sum_{i=1}^N \theta_i(s_{t+1},a'),
$$

where $\theta_j^{-}$ comes from the target network.[^2]
2. Minimizes the **quantile regression loss** between each predicted $\theta_i(s_t,a_t)$ and each target $y_j$:

$$
\mathcal{L}(\theta) = \frac{1}{N}\sum_{i=1}^N \sum_{j=1}^N \rho_{\tau_i}^\kappa \big( y_j - \theta_i(s_t,a_t)\big),
$$

where $\rho_{\tau}^\kappa$ is the **quantile Huber loss**.[^5][^2]

The **quantile Huber loss** combines:

- The **check function** of quantile regression

$$
\rho_\tau(z) = |\tau - \mathbb{I}[z<0]|\,|z|,
$$

which is asymmetric around zero (different penalty for under‑ and over‑estimation).
- The **Huber loss** to reduce sensitivity to outliers: squared error near zero, linear far away.

In practice, SB3‑contrib uses exactly this quantile Huber loss per pair of predicted/target quantiles.[SB3-QRDQN-docs — strong][^6]

### 3.3 Action selection and difference from DQN

- **Action selection**: QRDQN selects actions using the **mean of quantiles**:

$$
Q(s,a) \approx \frac{1}{N} \sum_{i=1}^N \theta_i(s,a), \quad a^* = \arg\max_a Q(s,a).
$$[^2]
- **Difference from DQN**:
    - DQN outputs a single scalar $Q(s,a)$ per action and uses squared TD error.
    - QRDQN outputs $N$ quantile values per action, trains with quantile Huber loss to match the entire target distribution, and uses the average of quantiles for greedy actions.
    - Conceptually, QRDQN learns a distribution over returns, not just the mean, enabling richer uncertainty and potential risk‑aware policies.

**Claim 5 (theoretical properties).** QR‑based distributional RL (QRDQN) can be seen as approximating the distributional Bellman operator under the 1‑Wasserstein metric; later work shows that the quantile loss and Cramér distance–based projections share minimizers and have desirable gradient properties.[Dabney2017-QRDQN; Lheritier2022-CramerQR — evidence: strong][^7][^2]
Caveat: The full convergence theory remains more complex than for classic Q‑learning; for your thesis, high‑level properties are sufficient.

***

## 4. Source matrix

| Citation key | Source | Year | Topic | Contribution | Why it matters | How to cite in thesis |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| Mnih2015-DQN | Mnih et al., “Human-level control through deep reinforcement learning,” *Nature* | 2015 | DQN foundations | Introduced DQN with CNN, replay, target network, $\epsilon$-greedy; first human‑level Atari results | Canonical reference for value‑based deep RL and baseline DQN structure | Use in RL background section to define Q‑learning, target network, replay buffer, and exploration[^1] |
| Bellemare2017-Distributional | Bellemare et al., “A Distributional Perspective on Reinforcement Learning,” arXiv | 2017 | Distributional RL, C51 | Formalizes value distribution $Z(s,a)$, defines distributional Bellman operator, introduces C51 algorithm | Theoretical foundation of distributional RL; shows distributional DQN variant can outperform DQN on Atari | Cite when motivating distributional RL and introducing C51 as precursor to QRDQN[^3] |
| Dabney2017-QRDQN | Dabney et al., “Distributional Reinforcement Learning with Quantile Regression,” AAAI | 2018 | QRDQN | Proposes QRDQN: quantile parameterization of return distribution, quantile Huber loss, Atari evaluation | Primary reference for QRDQN; shows significant gains over DQN and C51 on Atari | Core citation for your algorithm choice; use when explaining quantiles, loss function, and benefits over DQN[^2][^8][^9] |
| Dabney2018-IQN | Dabney et al., “Implicit Quantile Networks for Distributional RL,” ICML | 2018 | IQN | Extends QRDQN by sampling arbitrary quantile fractions, giving more flexible distributions and risk‑sensitive policies | Shows maturity of quantile‑based DRL and enables risk‑sensitive control | Cite briefly as more advanced distributional RL; you are not using IQN but can mention it in related work[^10][^11][^4] |
| Lheritier2022-CramerQR | Lhéritier \& Bondoux, “A Cramér Distance perspective on Quantile Regression based DRL,” PMLR | 2022 | Theory for QRDQN | Analyzes QRDQN under Cramér distance, clarifies why quantile regression loss works; proposes efficient projection algorithm | Underpins theoretical justification of quantile loss and non‑crossing constraints | Optional: cite in a short theory paragraph to show QRDQN isn’t an ad‑hoc hack[^7][^12][^13] |
| CEQRDQN2025 | “Uncertainty-Aware Deep RL with Calibrated Quantile Regression and Evidential Learning,” IEEE | 2025 | Uncertainty-aware QRDQN | Combines evidential learning with calibrated quantile regression in DQN, demonstrates uncertainty benefits | Shows distributional/quantile methods improve uncertainty estimates in safety‑critical robotics | Optional support for risk‑aware and uncertainty‑aware aspects of QRDQN[^14] |
| SB3-QRDQN | SB3‑contrib documentation: “QR-DQN — Stable Baselines3 - Contrib” | 2024+ | Implementation | Documents QRDQN implementation in SB3‑contrib: architecture, hyperparameters, supported envs | Directly relevant to your code; ensures your use of QRDQN matches published algorithm | Cite in implementation section when describing SB3‑contrib and hyperparameters[^6] |
| DRL-CyberSurvey2019 | “Deep Reinforcement Learning for Cyber Security”, arXiv survey | 2019 | DRL in cybersecurity | Surveys DRL for cyber defense, including RL‑based IDS, penetration testing, and defence games | Shows DRL (including DQN variants) has been applied to cyber defense; no specific QRDQN use noted | Cite in NIDS/cybersecurity background to position DRL, then note that distributional variants are still rare[^15] |
| DuelingDoubleDQN-IDS2022 | “Enabling intrusion detection systems with dueling double deep Q‑networks”, *Digital Threats* | 2022 | RL NIDS | Applies dueling double DQN to IDS, compares with ML baselines on older datasets | Example of advanced DQN variants used in IDS | Cite as NIDS RL precedent; contrast your QRDQN choice as another DQN family extension[^16] |
| QRDQN-Pentest2025 | Alhamed \& Rahman, “Automated DoS Penetration Testing Using Deep Q Learning,” IJACSA | 2025 | Cyber attack planning | Combines DQN and QRDQN to optimize attack paths in penetration testing | Shows QRDQN used in a cybersecurity context (offensive side) and compares with DQN | Use as evidence that QRDQN has been tried in security, but note it is attack‑path planning, not NIDS[^17] |


***

## 5. QRDQN in cybersecurity

**Claim 6 (state of QRDQN in security).** Distributional RL and QRDQN have seen limited but non‑zero use in cybersecurity, mostly in **attack/penetration planning** and **uncertainty‑aware control**, not in mainstream NIDS classification.

Findings from the search:

- **Penetration testing / attack path planning.**
    - Alhamed \& Rahman use DQN and QRDQN to generate optimal DoS attack paths, noting that QRDQN improves exploration and convergence at the cost of more resources.[^17]
    - Evidence strength: **moderate**; venue is applied and results are domain‑specific.
    - Relevance: shows QRDQN applied in cyber context, but on the attacker side (planning), not flow‑based IDS.
- **RL‑based IDS variants (non‑distributional).**
    - Several works use DQN, Double DQN, and Dueling DQN for IDS, usually on NSL‑KDD or CICIDS2017, but without modeling return distributions; e.g., DuelingDoubleDQN‑IDS applies dueling double DQN and compares with ML baselines.[^16][^18][^19]
    - Evidence strength: **moderate–strong**; supports the idea that DQN‑family methods are considered reasonable for IDS, but distributional variants are under‑explored.
- **DRL cyber defense surveys.**
    - Surveys of DRL for cyber security mention DQN, dueling DQN, Double DQN, A3C, etc., but do not highlight QRDQN or IQN as standard choices in NIDS.[^15][^19]

I did not find widely cited work where **QRDQN is directly used as a flow‑based NIDS classifier on CICIDS2017 or UNSW‑NB15**. Evidence for “QRDQN in NIDS” is therefore **weak to non‑existent** as of 2026, while DQN variants are present.

**Thesis implication:** You must present QRDQN as a **well‑established RL algorithm from the general RL literature** that you are applying experimentally to NIDS, not as an established standard in cybersecurity.

***

## 6. Justification for my thesis

A cautious, defensible justification:

1. **QRDQN is a robust, widely evaluated extension of DQN.**
    - Dabney et al. show that QRDQN significantly outperforms DQN and C51 on the full Atari benchmark, indicating that quantile‑based distributional RL is competitive and stable in large‑scale discrete‑action tasks.[^2]
    - Evidence strength: **strong**.
    - Thesis use: “We choose QRDQN as a mature, off‑policy deep RL algorithm that extends DQN with distributional modeling and has been extensively evaluated in standard RL benchmarks.”
2. **It naturally supports cost‑sensitive and risk‑aware decisions.**
    - By modeling a distribution over returns, QRDQN allows, at least conceptually, risk‑sensitive policies (e.g., focusing on low quantiles to avoid catastrophic outcomes), which is aligned with asymmetric costs in intrusion detection (false negatives more costly than false positives).[^4][^2]
    - Evidence strength: **moderate** (RL literature plus your cost model).
    - Thesis use: “In our NIDS setting, the distributional estimate could be exploited to design risk‑averse policies; in this thesis we primarily use the mean over quantiles but preserve the distribution for future work.”
3. **SB3‑contrib provides a reproducible implementation.**
    - Stable‑Baselines3’s contrib module implements QRDQN following Dabney et al., with documented hyperparameters and evaluation scripts.[^6]
    - Evidence strength: **strong**.
    - Thesis use: “We use the SB3‑contrib implementation of QRDQN to ensure reproducibility and adherence to the original algorithm; we report all hyperparameters and versions.”
4. **Its use here is exploratory, not claimed as state‑of‑the‑art in NIDS.**
    - Given that QRDQN is not standard in NIDS, you should explicitly state that your use is exploratory and that your primary question is: *“Does QRDQN offer any practical advantages over strong supervised baselines (Random Forest) and standard DQN on CICIDS2017 under rigorous evaluation?”*
    - This makes the contribution a **methodological comparison and risk‑aware RL exploration**, not a claim of superiority.
5. **Baselines and limitations are central.**
    - You must compare QRDQN against:
        - Random Forest (strong supervised baseline on tabular flow data).
        - Optionally, a simple DQN with the same architecture but scalar output.
    - You must also discuss stability (sensitivity to hyperparameters, seeds) and sample efficiency vs. RF, being willing to report negative or mixed results.

***

## 7. Hyperparameter explanation table

*(Aligned with SB3‑contrib QRDQN docs; you will set specific values in your experiments.)*


| Hyperparameter | Meaning | Effect | Risk if badly chosen | How to report it |
| :-- | :-- | :-- | :-- | :-- |
| `learning_rate` | Step size in optimizer (e.g., Adam) | Controls how fast network weights are updated | Too high → divergence or unstable Q estimates; too low → very slow learning | Report value (e.g., 1e‑4), schedule (constant/linear), and optimizer used; justify choice via reference or small ablation[^6] |
| `buffer_size` | Max number of transitions in replay buffer | Larger buffers give more diverse experience, stabilize updates | Too small → overfitting to recent transitions, instability; too large → higher memory, older data may be stale | Report size (e.g., 100,000), whether reservoir or FIFO; note that dataset is static so large buffers mostly replicate dataset |
| `learning_starts` | Number of environment steps before starting gradient updates | Allows buffer to fill with some diversity before training | Too small → training on tiny, unrepresentative buffer; too large → wasted computation | Report exact number (often few thousand steps) and explain it is fixed across runs |
| `batch_size` | Number of transitions per gradient update | Affects gradient variance and compute per update | Too small → noisy gradients; too large → slow updates, possible over‑smoothing | Report size (e.g., 64/128/256) and consider reusing standard SB3 defaults; mention any tuning you perform |
| `gamma` | Discount factor $\gamma\in(0,1]$ | Controls horizon of return; near 1 focuses on long‑term rewards | Too low → myopic policy; too high → harder credit assignment and higher variance | In your static classification, $\gamma$ near 1 is standard; report chosen value and rationale (e.g., 0.99) |
| `train_freq` | How often to train (in environment steps) | Smaller values → more frequent updates per interaction | Too frequent → overfitting to small changes; too sparse → slow learning | Report as `(train_freq, gradient_steps)` pair, per SB3 convention; clarify you keep total updates comparable across runs |
| `target_update_interval` | Steps between target network updates | Larger interval → more stable but slower to adapt | Too small → target oscillation; too large → stale targets, slow propagation | Report interval (e.g., every 1000 steps) and relate it to DQN defaults; mention if you experimented with alternative values |
| `exploration_fraction` | Fraction of total training steps over which $\epsilon$ decays | Determines how quickly exploration decreases | Too fast decay → premature exploitation; too slow → excessive random actions | Report value (e.g., 0.1); show schedule for $\epsilon$ from `initial` to `final` |
| `exploration_initial_eps` | Initial exploration rate $\epsilon_0$ | Higher → more random actions early | Too high with short training → inefficient; too low → insufficient exploration | Report value (e.g., 1.0 or 0.5) and justify via standard RL practice or data‑efficiency study |
| `exploration_final_eps` | Final exploration rate $\epsilon_{\min}$ | Ensures some residual exploration | Too high → noisy policy; too low → purely greedy, risky for non‑stationary envs | Report value (e.g., 0.01) and note that your environment is static offline; exploration mainly affects training, not evaluation |
| `policy network architecture` | Number of layers and units in MLP that predicts quantiles | Controls representational capacity and overfitting risk | Too small → underfitting complex decision boundary; too large → overfitting and unstable training | Report architecture (e.g., two FC layers with 256 units, ReLU), shared across DQN/QRDQN for fair comparison |
| `n_quantiles` | Number of quantile atoms per action | More quantiles → finer distribution approximation | Too few → coarse distribution; too many → more parameters, slower training, possible overfitting | Report value (e.g., 51 or 200 as in original paper/implementation); explain that action value is mean over quantiles; mention in limitations if you did not tune it |
| `max_grad_norm` | Gradient clipping threshold | Stabilizes training by limiting large updates | Too small → slow learning; too large or none → potential exploding gradients | Report value (e.g., 10.0) and whether gradient clipping is enabled |
| `seed` | Random seed | Controls initialization and sampling | Not varying seeds → results may be non‑robust | Report seeds used and number of runs; present averages and standard deviations |


***

## 8. Thesis-ready explanation

Suggested subsection: **“QRDQN and distributional reinforcement learning”**

You can structure it as:

1. **From DQN to distributional RL.**
2. **C51 and the value distribution.**
3. **QRDQN: quantile-based distributional DQN.**
4. **Why QRDQN is chosen in this work.**
5. **Limitations and evaluation strategy.**

Example paragraphs (in English; Codex will translate and adapt):

> **From DQN to distributional RL.**
> Classic Deep Q‑Networks approximate the expected return $Q(s,a)$ of taking action $a$ in state $s$, using a neural network trained to minimize a squared temporal‑difference error with respect to a Bellman target and stabilized through a target network and experience replay.[Mnih2015-DQN] This approach collapses all randomness in rewards and transitions into a single scalar estimate per state–action pair. Distributional reinforcement learning generalizes this idea by modeling the full **distribution** of returns $Z(s,a)$, rather than only its expectation, and defines a distributional Bellman operator that propagates return distributions instead of mean values.[Bellemare2017-Distributional][^3][^1]

> **C51 and the value distribution.**
> Bellemare et al. introduced C51, which approximates $Z(s,a)$ by a categorical distribution supported on a fixed set of 51 equally spaced atoms in a value interval, and trains a neural network to output probabilities over these atoms while minimizing a Kullback–Leibler divergence between target and predicted distributions.[Bellemare2017-Distributional] On the Atari benchmark, C51 achieves better performance than the original DQN, supporting the hypothesis that modeling the value distribution can provide a richer learning signal.[^3]

> **QRDQN: quantile-based distributional DQN.**
> Dabney et al. proposed Quantile Regression DQN (QRDQN), which parameterizes the return distribution $Z(s,a)$ using a set of quantiles rather than fixed categorical atoms.[Dabney2017-QRDQN] For each state–action pair, the network outputs $N$ scalar values $\{\theta_i(s,a)\}$, each representing the value of the return at a fixed quantile level $\tau_i$. Training minimizes a quantile Huber loss between these predicted quantiles and distributional Bellman targets built from the next state, effectively projecting arbitrary return distributions onto a staircase distribution that approximates the full quantile function.[Dabney2017-QRDQN; Lheritier2022-CramerQR] Actions are still selected greedily based on the mean of these quantiles, so QRDQN remains compatible with standard discrete‑action RL APIs while providing additional information about the variability and tail behavior of returns.[^7][^2]

> **Rationale for QRDQN in this thesis.**
> In this work, QRDQN is used as a distributional variant of DQN to learn a binary PERMIT/BLOCK policy over flow‑level observations. The choice is motivated by (i) its strong empirical performance and robustness on standard RL benchmarks, (ii) its ability to represent a distribution over returns, which in principle allows for risk‑sensitive decision rules compatible with asymmetric costs for false negatives and false positives, and (iii) the availability of a tested implementation in the SB3‑contrib library, which facilitates reproducible experiments and rigorous hyperparameter reporting.[Dabney2017-QRDQN; SB3-QRDQN-docs] We explicitly treat this use as exploratory and compare QRDQN against strong non‑RL baselines (e.g., Random Forest) as well as, when feasible, a standard DQN with scalar outputs.[^6][^2]

> **Limitations and scope.**
> It is important to emphasize that QRDQN is not yet a standard algorithm in intrusion detection systems; existing NIDS research predominantly uses non‑distributional DQN variants or purely supervised methods.[DRL-CyberSurvey2019; DuelingDoubleDQN-IDS2022] Moreover, in the static classification‑as‑RL formulation used here, the episodic structure is largely artificial and the benefit of modeling return distributions may be more limited than in strongly stochastic, sequential control tasks. For this reason, the thesis does not assume that QRDQN is universally better than classical methods; instead, it evaluates its performance empirically on CICIDS2017 under strict evaluation protocols and interprets any performance differences in light of algorithmic complexity, stability and cost‑sensitivity.[^15][^16]

***

## 9. Codex handoff

How Codex should use this dossier:

1. **Theoretical background section.**
    - In “Fundamentos de aprendizaje por refuerzo”, explain DQN (state, action, reward, Q‑valor, ecuación de Bellman, red objetivo, *replay buffer*, política $\epsilon$-codiciosa), citing Mnih2015-DQN.[^1]
    - In “Aprendizaje por refuerzo distribuicional”, introduce the idea of modelar la **distribución de retornos** $Z(s,a)$, referencing Bellemare2017-Distributional and explaining conceptually why expectations may be insufficient.[^3]
2. **QRDQN theory subsection (“QRDQN y aprendizaje por refuerzo distribuicional”).**
    - Use the paragraphs in section 8 as a template.
    - Include a simple mathematical description of quantiles, the quantile Huber loss, and action selection based on the mean of quantiles.
    - Cite Dabney2017-QRDQN as the main algorithm reference and optionally Lheritier2022-CramerQR for theoretical reinforcement.[^7][^2]
3. **Implementation explanation.**
    - In “Implementación con SB3‑contrib”, describe that you use the QRDQN implementation from Stable‑Baselines3‑contrib, listing all hyperparameters in a table as in section 7.
    - State explicitly that the **same network architecture and most hyperparameters** are used for any DQN baseline, to ensure fair comparison.
    - Cite SB3‑QRDQN-docs with a URL.[^6]
4. **Defense‑oriented explanation.**
    - Emphasize that QRDQN is chosen because it is a robust, well‑studied algorithm in general RL, not because it is already standard in NIDS.
    - Include a short paragraph noting that QRDQN has only been lightly explored in cybersecurity (e.g., attack‑path planning) and has not, to your knowledge, been systematically evaluated as a flow‑level NIDS classifier.[^17][^15]
    - Explicitly mention the need for **baseline comparisons** and the exploratory nature of this choice.
5. **Limitations section.**
    - Add 2–3 sentences in the “Limitaciones del enfoque de RL” subsection stating that:
        - Distributional RL increases model complexity and hyperparameter sensitivity.
        - In the static CICIDS2017 setting, the advantage of modeling return distributions is not guaranteed and must be demonstrated empirically.
        - QRDQN training may be slower and more resource‑intensive than RF or standard DQN.

If Codex follows these instructions, the thesis will present QRDQN as a **well‑grounded but exploratory algorithm choice**, technically justified by the RL literature, clearly compared to baselines, and appropriately caveated for the NIDS domain.
<span style="display:none">[^20][^21][^22][^23][^24][^25][^26]</span>

<div align="center">⁂</div>

[^1]: https://www.nature.com/articles/nature14236

[^2]: https://arxiv.org/abs/1710.10044

[^3]: https://arxiv.org/abs/1707.06887

[^4]: https://arxiv.org/abs/1806.06923

[^5]: https://xuance.org/documents/algorithms/drl/qrdqn.html

[^6]: https://sb3-contrib.readthedocs.io/en/master/modules/qrdqn.html

[^7]: https://arxiv.org/abs/2110.00535

[^8]: https://ojs.aaai.org/index.php/AAAI/article/view/11791

[^9]: https://www.semanticscholar.org/paper/Distributional-Reinforcement-Learning-with-Quantile-Dabney-Rowland/fe3e91e40a950c6b6601b8f0a641884774d949ae

[^10]: https://www.semanticscholar.org/paper/d85623ffae865f9ef386644dd02d0ea2d6a8c8de

[^11]: https://proceedings.mlr.press/v80/dabney18a.html

[^12]: https://proceedings.mlr.press/v151/lheritier22a.html

[^13]: https://www.semanticscholar.org/paper/2cdbddb14304434aef9fdb3d22e04fb89a742330

[^14]: https://ieeexplore.ieee.org/document/11127742/

[^15]: https://ar5iv.labs.arxiv.org/html/1906.05799

[^16]: https://www.emerald.com/dts/article/1/1/115/52701/Enabling-intrusion-detection-systems-with-dueling

[^17]: https://thesai.org/Publications/ViewPaper?Volume=16\&Issue=3\&Code=IJACSA\&SerialNo=95

[^18]: https://deep.ai/publication/deep-q-learning-based-reinforcement-learning-approach-for-network-intrusion-detection

[^19]: https://arxiv.org/html/2410.07612v2

[^20]: https://www.semanticscholar.org/paper/67e07e0064f88503ded910ebf693b9c476209e07

[^21]: https://arxiv.org/abs/2305.16877

[^22]: https://www.semanticscholar.org/paper/8fa167d0db69e90b376b608acf534a640ff3d870

[^23]: https://www.semanticscholar.org/paper/9f873fecd24e0cfb800249a30ecd8e3b3155e709

[^24]: https://medium.com/@khalil.hennara.247/distributional-perspective-on-reinforcement-learning-c51-222c92dc5bca

[^25]: https://arxiv.org/abs/2603.03502

[^26]: https://www.semanticscholar.org/paper/A-Deep-Learning-based-Penetration-Testing-Framework-Koroniotis-Moustafa/b59b1351ad6803a3dfc40ff2a86d345d153c6f65

