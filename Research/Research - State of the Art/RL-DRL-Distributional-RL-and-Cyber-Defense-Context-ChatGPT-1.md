# Reinforcement Learning State of the Art for an RL-Based Cybersecurity Defender

## Research synthesis and main takeaways

For the reinforcement-learning side of your thesis, the literature supports a careful but useful framing. Reinforcement learning is natively a framework for **sequential** decision-making in which an agent interacts with a dynamic environment, observes state, selects actions, and optimizes long-term return. That framing is fully natural for network defense problems in which defender actions alter future system state, attacker opportunities, and future rewards. It is only **partially** natural for a static labeled-flow dataset formulation, where actions usually do not affect future observations and the environment does not react. In that case, the formulation can still be legitimate as an experimental design, but it should be positioned more modestly: as a cost-sensitive, sequentially processed benchmark inspired by RL, rather than as a full autonomous cyber-defense environment. citeturn1search4turn27search18turn26search6turn1search1turn28search0

For the chapter itself, the strongest research-backed storyline is the following. First, motivate RL from MDPs, policies, value functions, and the distinction between episodic and continuing tasks. Second, explain why tabular methods are inadequate for high-dimensional flow features and why DQN-style function approximation became the standard entry point for value-based deep RL. Third, explain distributional RL as a change in **what is estimated**: not only the mean return, but an approximation to the distribution of returns. Fourth, state clearly that this does **not** automatically yield safety guarantees, calibrated uncertainty, or superiority in NIDS; it only creates the possibility of using richer return information in decision-making. Fifth, distinguish genuine sequential cyber-defense RL from “classification-as-RL” papers that train on fixed labeled datasets. That distinction is especially important for a bachelor thesis, because it keeps your claims accurate and defensible. citeturn22search0turn22search1turn22search2turn16search2turn32search10

A second strong takeaway is that the cybersecurity literature is heterogeneous. There is now a sizeable body of work and several surveys on RL for IDS, network defense, SDN security, anomaly detection, and autonomous cyber defense. However, much of the RL-for-IDS literature still evaluates on fixed benchmark datasets, often with supervised labels driving reward design. In parallel, another branch of the literature studies **truly interactive** cyber environments such as Bayesian-attack-graph / POMDP formulations, CybORG, and CyberBattleSim-like settings, where defender or attacker actions change future states and therefore better match the assumptions of sequential control. The thesis should explicitly place itself in the first category, while referencing the second category as the more faithful long-term target for autonomous cyber defense. citeturn29search6turn29search3turn30search0turn32search10turn32search12

## Reinforcement learning foundations relevant to this thesis

In standard RL, the agent–environment interaction is modeled through states, actions, rewards, and transitions. When the problem satisfies the Markov property, it can be formalized as a Markov decision process: decisions depend on the current state, actions influence both immediate reward and future state, and a policy maps states to action probabilities. Value functions measure expected return under a policy, either from a state or from a state–action pair, and they are the organizational core of classical RL algorithms. This is the right conceptual starting point for your chapter, because it lets you define the thesis environment precisely and then explain where that formulation matches, and where it departs from, standard sequential RL. citeturn1search4turn27search18turn27search0turn27search14

The episodic versus continuing distinction matters for thesis positioning. Episodic tasks have terminal states and can be broken into separate episodes; continuing tasks do not naturally terminate and are often framed through discounted return or average reward. In cybersecurity, both framings appear in the literature. A multi-stage incident-response scenario over an attack window can be described episodically, while ongoing traffic monitoring and infrastructure protection are closer to continuing tasks. In a dataset-as-environment formulation, the “episode” is often an imposed convenience rather than a naturally reactive process, so it is worth saying explicitly whether an episode corresponds to a file, a traffic trace, a shuffled batch, or an operational time window. citeturn26search6turn26search1turn26search25

The standard model-free / model-based distinction is also directly relevant. Model-free RL learns a controller or value estimates without explicitly learning environment dynamics. Model-based RL learns or uses a model of transitions and rewards, then plans with it. In modern cyber-defense discussions, model-based formulations often appear through attack graphs, POMDPs, or simulated network environments, whereas many deep Q-learning approaches for IDS are model-free. Since your implementation uses QRDQN on a Gymnasium-style benchmark, it sits in the **model-free, value-based** corner of the RL landscape. That is a clean and defensible classification. citeturn23search3turn23search9turn29search6

The on-policy / off-policy distinction is equally important for methodological clarity. Sutton and Barto’s canonical exposition labels Sarsa as an **on-policy TD control** method and Q-learning as an **off-policy TD control** method; the latter learns about a greedy target policy while behavior may remain exploratory. DQN and its descendants inherit this off-policy lineage, which is why replay buffers make sense for them: they can learn from past transitions that were not generated by the current network weights. This matters for your chapter because QRDQN should be introduced not just as “a deep network,” but more precisely as an **off-policy, value-based, distributional extension of DQN**. citeturn25search0turn25search1turn24search3

Tabular RL is inadequate once states are high-dimensional feature vectors or continuous spaces. Classic Q-learning assumes a discrete representation in which values can be stored per state or state–action pair; deep RL arose precisely because many interesting control problems have high-dimensional inputs for which a lookup-table representation is infeasible. Policy-gradient work also emphasized that function approximation becomes essential in realistic settings, and DQN’s breakthrough was to learn approximate action values directly from high-dimensional sensory inputs. For a flow-based IDS with dozens of numerical features, this is the right justification for leaving tabular RL behind. citeturn18search0turn19search0turn2search13

The deepest conceptual limitation in your setting is the use of a static labeled dataset. Offline RL literature stresses that learning from previously collected data is fundamentally harder because the learner cannot ask the environment for new trajectories, and standard off-policy methods can fail under distribution shift or extrapolation error. In intrusion-detection papers, this is sometimes addressed explicitly by describing the task as “supervised problems” or “offline reinforcement learning.” The important thesis-level point is that when each flow is labeled independently, and the chosen action does not alter future observations, the sequential-control interpretation is weakened: the problem begins to resemble cost-sensitive classification or contextual bandit-style decision-making more than full closed-loop cyber defense. That does **not** invalidate your benchmark, but it sharply constrains what you can claim from it. citeturn1search1turn1search2turn28search0turn28search3

## Deep and distributional reinforcement learning

DQN is the classic bridge from tabular Q-learning to deep RL. The 2013 Atari paper showed that a neural network could approximate action values directly from high-dimensional input, and the 2015 Nature paper established the deep Q-network as the canonical value-based deep RL algorithm. Two design choices became central to its stability: a replay memory for sampling prior transitions, and a separate target network for computing lagged bootstrap targets. These are not implementation details you can safely omit; they are part of why DQN became the foundational baseline for value-based DRL. citeturn19search0turn19search4turn20search14

Experience replay and target networks deserve explicit explanation in your state-of-the-art text because they motivate later DQN variants. Replay lets the learner reuse past transitions rather than updating only from the newest observation, while the target network reduces the instability that would arise if the same moving network were used simultaneously for prediction and target construction. Double DQN was proposed because vanilla DQN can overestimate action values; its key idea is to decouple action selection from action evaluation in the bootstrap step. This is directly relevant when framing QRDQN, because QRDQN is not a different philosophical family from DQN but a later point in the same value-based lineage. citeturn19search4turn20search14turn19search3

Prioritized Experience Replay further modified the DQN recipe by sampling transitions according to learning utility rather than uniformly, and Rainbow later showed that several DQN improvements can be fruitfully combined into a single agent that outperforms older baselines on Atari. Rainbow is therefore relevant in your chapter as context, but it should not be presented as mandatory background for understanding your implementation. A compact positioning sentence is enough: Rainbow is best understood as an aggregated DQN family member, whereas QRDQN is one specific distributional branch that changes the learning target from expected action value to return distribution. citeturn0search3turn0search7turn2search7turn2search3

Actor–critic methods should be treated as background only, as you requested. Their basic idea is to combine a parameterized policy (the actor) with a learned value estimator (the critic). Foundational work framed actor–critic algorithms as two-time-scale methods in which the critic performs TD-style evaluation and the actor updates policy parameters in an approximate gradient direction; later policy-gradient work and PPO-style methods made that design family highly influential in modern RL. For your thesis, however, the function of actor–critic discussion is comparative: it tells the reader that deep RL includes both value-based and policy-based lines of work, while making clear that your method belongs to the value-based line derived from DQN rather than to the actor–critic line. citeturn18search1turn18search5turn18search0turn18search2

Distributional RL changes the object being estimated. Bellemare, Dabney, and Munos argued that the value distribution—the distribution of random return induced by stochastic rewards, transitions, and policy choices—is fundamentally richer than the traditional scalar expectation of return. C51 approximates this return distribution on a fixed categorical support, while QR-DQN replaces the fixed support with learned quantiles via quantile regression. QR-DQN’s original paper reported strong Atari performance and highlighted an advantage over C51: it does not require predefined support bounds, and the number of quantiles controls the granularity of the approximation. citeturn22search0turn22search1turn22search5

For cybersecurity, the intuitive appeal of distributional RL is not hard to justify, but the wording must stay precise. Security actions often have **asymmetric consequences**: a false negative on a high-impact attack can be much worse than a false positive, and the mean return may hide rare but very costly outcomes. Because distributional RL approximates a return distribution rather than only its mean, it can in principle support risk-sensitive decision rules based on tails, quantiles, or CVaR-like criteria. However, the literature is equally clear that learning a return distribution does **not** by itself make the policy risk-sensitive. Risk sensitivity requires an explicit decision criterion extracted from that distribution; otherwise the agent may still act by maximizing a mean summary. Distributional RL also does not by itself prove calibrated uncertainty, operational safety, or superiority on a new domain such as NIDS. citeturn22search0turn22search2turn22search6turn22search10turn22search15

The evidence for distributional RL specifically in cybersecurity is real but still limited. In this research pass, the clearest **peer-reviewed** example I found is Benaddi et al. (Sensors, 2022), which uses a distributional RL-based IDS together with GAN-based data augmentation for industrial IoT anomaly detection. Recent surveys and secondary sources also cite a 2023 IEEE Access paper titled *C51-RL-IDS: Categorical RL for intrusion detection*, which suggests that categorical distributional RL has already entered the IDS literature. By contrast, strong peer-reviewed evidence for **QR-DQN specifically in IDS/NIDS or autonomous cyber defense** appears much sparser. The QR-DQN examples I could locate were more visible in adjacent or weaker-evidence contexts such as offensive DoS penetration testing and phishing-detection preprints, not in a mature NIDS literature comparable to mainstream DQN/DDQN/PPO-style work. That makes QRDQN in your thesis a plausible exploratory choice, but not one that can be presented as already established best practice in NIDS. citeturn13search1turn11search1turn11search3turn7search0turn11search2

## RL for cybersecurity and what counts as genuine sequential defense

The RL-for-cybersecurity literature is now broad enough to cite as a real field rather than as a niche curiosity. Surveys from 2022 to 2026 review RL applications in intrusion detection, network defense, SDN security, IoT/IIoT, DDoS mitigation, attack-path reasoning, and autonomous cyber defense. A recent IEEE Communications Surveys & Tutorials review specifically addresses RL-based intrusion detection in communication networks, while newer survey work also emphasizes the gap between promising benchmark results and realistic deployment conditions. This gives you solid support for a literature section on “RL for cybersecurity,” but it also implies that you should differentiate **subfields** rather than treat all cyber RL papers as methodologically comparable. citeturn32search0turn32search10turn32search12turn16search2

A useful distinction is between **intrusion detection on fixed datasets** and **dynamic cyber defense in interactive environments**. The first category includes papers that apply DQN-like or related RL algorithms to labeled traffic datasets, often with rewards derived from correctness, class imbalance, or misclassification cost. Some works say this explicitly: López-Martín et al. describe their contribution as applying deep reinforcement learning to intrusion detection for “supervised problems,” and later work from the same line speaks of “offline reinforcement learning.” This literature is relevant to your thesis because it is close to your benchmark style. But it should not be conflated with defender control in a reactive networked environment. citeturn28search0turn28search4turn28search3turn28search6

The second category is much closer to genuine sequential cyber defense. Hu, Zhu, and Liu model adaptive cyber defense against multi-stage attacks as a POMDP on Bayesian attack graphs, explicitly targeting autonomous defensive action under uncertainty. CybORG was introduced as a research gym for autonomous cyber agents with simulation and emulation modes, intended to support both blue-team and red-team decision making. Microsoft’s CyberBattleSim project similarly studies how automated agents interact in simulated enterprise networks, and later work extended it toward multi-agent attacker–defender training. These works are conceptually closer to autonomous cyber defense because actions alter the evolving cyber state and thereby change future observations and future reward. citeturn29search6turn29search3turn5search1turn30search0

This distinction leads directly to a criterion you can use in your chapter: **true sequential cyber-defense RL** requires that defender actions influence future environment state, attacker opportunity, or future observations. If blocking, quarantining, patching, shuffling, or reconfiguring a network alters the future attack surface, then RL is solving a sequential control problem. If, by contrast, each labeled flow is evaluated independently and the next sample is unaffected by the current action, then the formulation is better described as classification-as-RL, cost-sensitive classification, or at most a contextual decision process over a logged dataset. Many IDS papers nonetheless report strong results in that latter regime, so the distinction is methodological—not dismissive. citeturn1search4turn17search2turn28search0turn29search6

This is also where external-validity and data-shift concerns become decisive. Multiple NIDS critiques and cross-evaluation studies warn that near-perfect single-dataset results often do not survive deployment or cross-domain testing. Apruzzese et al. emphasize the dependence of ML-NIDS on costly labeled data and recurring reuse of the same datasets, while Arp et al. document broader pitfalls in ML-for-security evaluation, including sampling bias, unrealistic class balance, poor metrics, and base-rate-related problems. This means that an RL-based IDS outperforming a benchmark on CICIDS2017 is interesting, but not enough to claim operational superiority or broad generalization. That caution especially applies when the RL environment is built directly from a fixed labeled dataset. citeturn31search12turn31search2turn31search6turn31search13

## Positioning guidance for the bachelor thesis

The safest and strongest positioning is to describe QRDQN as an **exploratory algorithmic choice** inside a reproducible RL benchmark, not as an established best method for NIDS. You can justify the choice on three grounds. First, the problem uses moderately high-dimensional flow features, making deep value approximation more natural than tabular RL. Second, the action space is discrete and small, which makes DQN-family methods a natural fit. Third, distributional RL offers a richer representation of return than scalar Q-learning, which is appealing when the reward function encodes asymmetric false-positive and false-negative costs. All three statements are supportable from the RL and cyber-defense literature without overstating what the thesis proves. citeturn19search0turn22search0turn22search1turn16search2

At the same time, several claims should be avoided. The thesis should not present itself as demonstrating autonomous cyber defense in the strong sense used by POMDP or cyber-range papers, because the environment is ultimately based on logged data rather than a reactive network. It should not claim that QRDQN provides safety guarantees or calibrated uncertainty merely because it is distributional. It should not claim that RL is inherently more suitable than supervised learning for CICIDS2017-like benchmarks. And it should not imply that results on a single dataset settle the usefulness of RL for operational NIDS. Those claims would overshoot the literature. citeturn22search0turn22search2turn29search6turn31search12turn31search2

Supervised baselines remain mandatory, not optional. In your setting, they are needed for at least three reasons. First, if the environment is built from a static labeled dataset, much of the predictive signal is still fundamentally supervised. Second, strong supervised baselines—especially tree ensembles such as Random Forest—often perform very well on tabular intrusion datasets and therefore provide the correct yardstick for deciding whether the RL framing adds anything. Third, NIDS evaluation literature repeatedly warns that model choice must be separated from dataset artifacts and reward-design choices; comparing against competent supervised baselines is one of the simplest ways to reduce overclaiming. In other words, the presence of an RL agent does not remove the scientific obligation to ask whether a simpler classifier solves the benchmark equally well or better. citeturn28search0turn28search3turn31search12turn31search2

A good bachelor-thesis formula is therefore: *this work studies an RL-based, cost-sensitive network-flow defender in a dataset-as-environment benchmark, using QRDQN as a value-based distributional RL method and supervised learning models as necessary baselines; the goal is not to prove that RL or QRDQN is universally superior for NIDS, but to evaluate whether this formulation is technically coherent, reproducible, and empirically competitive under the chosen benchmark conditions*. That wording is ambitious enough to show contribution, but conservative enough to survive viva-style scrutiny. citeturn16search2turn32search10turn31search12

## Candidate paragraphs for integration

**RL foundations paragraph.** Reinforcement learning formalizes decision making as an interaction between an agent and an environment, typically modeled as a Markov decision process in which the current state summarizes the information needed for action selection. A policy specifies which action to take in each state, while state-value and action-value functions quantify expected long-term return under a policy. This framework is fundamentally sequential: actions are relevant not only because of their immediate reward, but because they alter subsequent states and therefore future return. This sequential aspect distinguishes reinforcement learning from conventional supervised classification and is the central reason why RL is attractive for adaptive network-defense problems in which present defensive decisions can shape future attack opportunities and observations. citeturn1search4turn27search18turn27search0turn27search14

**DQN paragraph.** When the observation space is high-dimensional, tabular RL becomes impractical because storing and updating values for every possible state–action pair is infeasible. Deep Q-learning addresses this issue by replacing the lookup table with a neural approximation of the action-value function. The DQN family also introduced key stabilizing mechanisms, notably replay memory and a separate target network, which made deep value-based learning substantially more robust. Subsequent variants refined this basic design: Double DQN mitigates value overestimation, prioritized replay biases sampling toward more informative transitions, and Rainbow combines several improvements into a single value-based agent. These developments form the immediate algorithmic background for QRDQN. citeturn18search0turn19search4turn20search14turn19search3turn0search7turn2search7

**Distributional RL paragraph.** Classical value-based RL estimates the expected return associated with each action, whereas distributional RL seeks to approximate the full distribution of possible returns. This change is conceptually important because the mean return may hide heavy tails or rare but severe outcomes. C51 represents the return distribution on a fixed categorical support, while QR-DQN uses quantile regression to learn a quantile-based approximation without requiring predefined support bounds. For security-oriented reward functions, this richer return representation is potentially attractive because defensive errors can have highly asymmetric consequences. However, distributional RL alone does not imply calibrated uncertainty, risk-sensitive behavior, or operational safety; such properties require additional design choices in the reward function, action-selection rule, or evaluation methodology. citeturn22search0turn22search1turn22search5turn22search2turn22search10

**Cybersecurity paragraph.** Reinforcement learning has been applied to intrusion detection, DDoS mitigation, SDN security, industrial anomaly detection, and autonomous cyber defense, but the literature spans substantially different problem formulations. One branch uses labeled traffic datasets and converts detection into a reward-driven learning problem, sometimes explicitly described as supervised or offline reinforcement learning. Another branch focuses on interactive environments, including attack-graph and POMDP models, cyber ranges, and gym-style simulators such as CybORG and CyberBattleSim, where agent actions truly affect future states. This distinction is methodologically important: the former literature is closer to cost-sensitive detection on logged data, whereas the latter is closer to genuine sequential cyber defense. citeturn28search0turn28search3turn13search1turn29search6turn29search3turn30search0

**Positioning paragraph.** The present thesis should therefore be positioned as a controlled experimental study of an RL-based defender over network-flow observations, rather than as a demonstration of fully autonomous cyber defense. In this framing, QRDQN is a justified exploratory choice because it is an off-policy, value-based deep RL algorithm suitable for discrete actions and high-dimensional observations, and because its distributional formulation may better expose asymmetric return structure than scalar Q-learning. Nevertheless, any claim of superiority over conventional methods must remain empirical and benchmark-specific. Strong supervised baselines are indispensable, both because the benchmark is ultimately derived from a labeled dataset and because prior NIDS literature has shown that single-dataset evaluations can overestimate real-world utility and generalization. citeturn25search1turn22search1turn31search12turn31search2

## Claims that are safe vs unsafe

**Safe claims**

- *This thesis uses a reinforcement-learning formulation over network-flow observations, but it is evaluated on a static labeled dataset rather than on a reactive cyber-range or live network environment.* This is accurate and conceptually important because offline or logged-data learning differs from interactive RL. citeturn1search1turn28search0turn28search3
- *QRDQN is a reasonable exploratory algorithm for this benchmark because it is an off-policy, value-based deep RL method for discrete actions, and distributional RL can represent richer return structure than mean-only Q-learning.* citeturn25search1turn22search0turn22search1
- *RL for intrusion detection and network defense is an active literature, but the field mixes dataset-based detection papers and truly sequential cyber-defense environments.* citeturn32search10turn29search6turn29search3turn30search0
- *Distributional RL is potentially attractive in security because security rewards may be asymmetric and tail outcomes may matter, but that is a motivation, not a theorem.* citeturn22search0turn22search2turn22search10
- *Supervised baselines such as Random Forest remain mandatory in this thesis because the benchmark is derived from labeled data and because NIDS evaluation literature warns against overinterpreting single-dataset results.* citeturn28search0turn31search12turn31search2

**Unsafe claims**

- *“QRDQN is the state of the art for NIDS”* is unsafe. I found limited cybersecurity evidence for distributional RL overall, and much weaker evidence for QR-DQN specifically in mainstream IDS/NIDS literature than for more conventional DRL variants. citeturn13search1turn11search3turn7search0turn11search2
- *“Distributional RL provides uncertainty estimates and therefore makes the system safe”* is unsafe. The literature supports richer return modeling, not automatic calibration or safety guarantees. citeturn22search0turn22search2turn22search10
- *“The thesis demonstrates autonomous cyber defense”* is unsafe unless the environment reacts to the defender’s actions and alters future state accordingly, as in POMDP or cyber-range formulations. citeturn29search6turn29search3turn30search0
- *“RL is inherently better than supervised learning for CICIDS2017-style detection”* is unsafe. That is an empirical question, and the NIDS literature shows that dataset choice and evaluation protocol strongly affect conclusions. citeturn31search12turn31search2
- *“Good results on CICIDS2017 imply deployment readiness”* is unsafe. Cross-evaluation and ML-security pitfalls literature argue strongly against that inference. citeturn31search12turn31search6turn31search13

## References to add to references.bib

Based on your uploaded `references.bib`, the core RL references already present include Sutton & Barto, DQN, Double DQN, Dueling DQN, Bellemare’s distributional RL paper, Dabney’s QR-DQN paper, Hu et al. on learning-based POMDP cyber defense, and several DRL-for-NIDS surveys. I therefore do **not** repeat those. The most important **correction** to an existing entry is Wanrong Yang et al.’s survey, which now has final publication metadata in *Applied AI Letters* rather than only preprint metadata. citeturn16search0turn16search1turn16search2

**Correction to an existing entry**

Use this to replace or strengthen your current `Yang2024DRLNIDSSurvey` entry. citeturn16search0turn16search1

```bibtex
@article{Yang2026DRLNIDSurvey,
  author  = {Yang, Wanrong and Acuto, Alberto and Zhou, Yihang and Wojtczak, Dominik},
  title   = {A Survey for Deep Reinforcement Learning Based Network Intrusion Detection},
  journal = {Applied AI Letters},
  volume  = {7},
  number  = {2},
  pages   = {e70026},
  year    = {2026},
  doi     = {10.1002/ail2.70026}
}
```

**Foundational RL sources missing from your current `.bib`**

Watkins and Dayan (1992) is the canonical Q-learning reference and is especially useful if you want to define Q-learning explicitly as model-free, off-policy TD control. citeturn24search3turn25search1

```bibtex
@article{WatkinsDayan1992QLearning,
  author  = {Watkins, Christopher J. C. H. and Dayan, Peter},
  title   = {Q-learning},
  journal = {Machine Learning},
  volume  = {8},
  number  = {3},
  pages   = {279--292},
  year    = {1992},
  doi     = {10.1007/BF00992698}
}
```

Lin (1992) is worth adding because it is the classic early source for experience replay. citeturn2search14turn2search6

```bibtex
@article{Lin1992ReactiveAgents,
  author  = {Lin, Long-Ji},
  title   = {Self-Improving Reactive Agents Based on Reinforcement Learning, Planning and Teaching},
  journal = {Machine Learning},
  volume  = {8},
  number  = {3},
  pages   = {293--321},
  year    = {1992},
  doi     = {10.1007/BF00992699}
}
```

Schaul et al. (ICLR 2016) is the standard prioritized replay reference. I did not include a DOI because I did not verify one from a publisher page during this pass. citeturn0search3turn0search7

```bibtex
@inproceedings{Schaul2016PER,
  author       = {Schaul, Tom and Quan, John and Antonoglou, Ioannis and Silver, David},
  title        = {Prioritized Experience Replay},
  booktitle    = {International Conference on Learning Representations},
  year         = {2016},
  eprint       = {1511.05952},
  archiveprefix= {arXiv},
  note         = {Conference paper at ICLR 2016}
}
```

Hessel et al. (2018) is the standard Rainbow reference and is useful if you want a compact sentence locating QRDQN within the broader DQN family. citeturn2search7turn2search3

```bibtex
@inproceedings{Hessel2018Rainbow,
  author    = {Hessel, Matteo and Modayil, Joseph and van Hasselt, Hado and Schaul, Tom and Ostrovski, Georg and Dabney, Will and Horgan, Dan and Piot, Bilal and Azar, Mohammad Gheshlaghi and Silver, David},
  title     = {Rainbow: Combining Improvements in Deep Reinforcement Learning},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume    = {32},
  number    = {1},
  year      = {2018},
  url       = {https://ojs.aaai.org/index.php/AAAI/article/view/11796}
}
```

Konda and Tsitsiklis (2003) is a strong background reference if you want one foundational actor–critic citation without expanding that branch too much. I did not include a DOI because I did not verify one from a publisher page during this pass. citeturn18search17turn18search5

```bibtex
@article{KondaTsitsiklis2003ActorCritic,
  author  = {Konda, Vijay R. and Tsitsiklis, John N.},
  title   = {On Actor-Critic Algorithms},
  journal = {SIAM Journal on Control and Optimization},
  volume  = {42},
  number  = {4},
  pages   = {1143--1166},
  year    = {2003}
}
```

**Offline RL references that directly support your methodological caveats**

Levine et al. (2020) is the most useful high-level reference for explaining why learning from static logged data is not equivalent to standard interactive RL. This is a **preprint**, and it should be marked as such. citeturn1search1

```bibtex
@article{Levine2020OfflineRL,
  author        = {Levine, Sergey and Kumar, Aviral and Tucker, George and Fu, Justin},
  title         = {Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems},
  journal       = {arXiv preprint arXiv:2005.01643},
  year          = {2020},
  eprint        = {2005.01643},
  archiveprefix = {arXiv},
  primaryclass  = {cs.LG},
  note          = {Preprint}
}
```

Fujimoto, Meger, and Precup (ICML 2019) is valuable if you want a concrete citation for extrapolation error and why naive off-policy deep RL struggles on fixed datasets. citeturn1search2turn1search6

```bibtex
@inproceedings{Fujimoto2019BCQ,
  author    = {Fujimoto, Scott and Meger, David and Precup, Doina},
  title     = {Off-Policy Deep Reinforcement Learning without Exploration},
  booktitle = {Proceedings of the 36th International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {97},
  pages     = {2052--2062},
  year      = {2019},
  url       = {https://proceedings.mlr.press/v97/fujimoto19a.html}
}
```

**Cybersecurity survey references missing from your current `.bib`**

Adawadkar and Kulkarni (2022) is a concise survey on RL in cybersecurity and is useful for a general “RL for cyber” framing. citeturn32search0turn32search14

```bibtex
@article{Adawadkar2022CyberSecurityRLSurvey,
  author  = {Adawadkar, Amrin Maria Khan and Kulkarni, Nilima},
  title   = {Cyber-security and Reinforcement Learning---A Brief Survey},
  journal = {Engineering Applications of Artificial Intelligence},
  volume  = {114},
  pages   = {105116},
  year    = {2022},
  doi     = {10.1016/j.engappai.2022.105116}
}
```

Kheddar et al. (2024) is the strongest missing **review article** for the IDS side and fits your chapter extremely well. citeturn32search10turn32search7

```bibtex
@article{Kheddar2024RLIDSReview,
  author  = {Kheddar, Hamza and Dawoud, Diana W. and Awad, Ali Ismail and Himeur, Yassine and Khan, Muhammad Khurram},
  title   = {Reinforcement-Learning-Based Intrusion Detection in Communication Networks: A Review},
  journal = {IEEE Communications Surveys \& Tutorials},
  volume  = {27},
  number  = {4},
  pages   = {2420--2469},
  year    = {2024},
  doi     = {10.1109/COMST.2024.3484491}
}
```

Javadpour et al. (2026) is broader than IDS alone and is useful if you want a modern umbrella citation for RL in network security, including defense and environment design. citeturn32search12turn32search9

```bibtex
@article{Javadpour2026BeyondRLNetworkSecurity,
  author  = {Javadpour, Amir and Ja'fari, Forough and Taleb, Tarik and Turkmen, Fatih and Benzaid, Chafika},
  title   = {Beyond Reinforcement Learning for Network Security: A Comprehensive Survey and Tutorial},
  journal = {Journal of Information Security and Applications},
  volume  = {96},
  pages   = {104294},
  year    = {2026},
  doi     = {10.1016/j.jisa.2025.104294}
}
```

**Interactive cyber-environment references that strengthen the “future work / stronger RL setting” discussion**

Standen et al. (2021) is useful for contrasting your dataset-based benchmark with a more realistic autonomous-cyber-agent environment. This is a **preprint / workshop-era** reference, so mark it clearly as such. citeturn29search3turn29search7turn29search19

```bibtex
@article{Standen2021CybORG,
  author        = {Standen, Maxwell and Lucas, Martin and Bowman, David and Richer, Toby J. and Kim, Junae and Marriott, Damian},
  title         = {CybORG: A Gym for the Development of Autonomous Cyber Agents},
  journal       = {arXiv preprint arXiv:2108.09118},
  year          = {2021},
  eprint        = {2108.09118},
  archiveprefix = {arXiv},
  primaryclass  = {cs.CR},
  note          = {Preprint; also cited as an IJCAI-21 Adaptive Cyber Defense workshop paper}
}
```

## Open questions and limitations

The weakest part of the evidence base remains **QR-DQN specifically in NIDS/cyber-defense**. In this pass, I found clear distributional-RL evidence in industrial IoT anomaly detection and secondary evidence for a C51-based IDS paper, but not a strong, mature, peer-reviewed QR-DQN-in-NIDS literature comparable in visibility to mainstream DQN/DDQN/PPO-style cyber papers. That gap is actually useful for your thesis positioning: it supports presenting QRDQN as a credible but exploratory choice. It does **not** support presenting it as established NIDS state of the art. citeturn13search1turn11search3turn7search0turn11search2