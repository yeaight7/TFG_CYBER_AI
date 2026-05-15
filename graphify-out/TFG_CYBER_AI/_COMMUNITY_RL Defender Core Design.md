---
type: community
members: 60
---

# RL Defender Core Design

**Members:** 60 nodes

## Members
- [[152-Dimensional Observation Vector_1]] - rationale - docs/Personal Research/data-structure-and-canonical-schema-research-report.md
- [[Adam epsilon coupled to batch_size (eps=0.01batch_size)]] - rationale - docs/Personal Research/qrdqn-research-report.md
- [[Anti-Leakage Policy (drop IPs, timestamps, Flow IDs, ports)]] - rationale - docs/Personal Research/deep-defense-research/02-datos-esquema-canonico-y-preprocesado.md
- [[Asymmetric Cost Reward Design (FN  FP penalty)]] - rationale - docs/Personal Research/data-structure-and-canonical-schema-research-report.md
- [[Best Run C03 (accuracy=0.99859, random split)]] - document - docs/Personal Research/data-structure-and-canonical-schema-research-report.md
- [[Binary Action Space PERMIT(0)  BLOCK(1)]] - rationale - docs/Personal Research/deep-defense-research/01-fundamentos-y-objetivo.md
- [[C51 (Categorical DQN, distributional RL predecessor)]] - rationale - docs/Personal Research/qrdqn-research-report.md
- [[CICIDS2017 Dataset (8 CSVs, 5-day capture)]] - document - docs/Personal Research/deep-defense-research/02-datos-esquema-canonico-y-preprocesado.md
- [[Canonical Schema (76 Flow Features)]] - rationale - docs/Personal Research/data-structure-and-canonical-schema-research-report.md
- [[Check A Direct Evaluation (anti-env-dependency)]] - rationale - docs/Personal Research/data-structure-and-canonical-schema-research-report.md
- [[Check B Shuffled Labels Anti-Leakage Test]] - rationale - docs/Personal Research/data-structure-and-canonical-schema-research-report.md
- [[Check C Hard CSVDay Split Generalization Test]] - rationale - docs/Personal Research/data-structure-and-canonical-schema-research-report.md
- [[Classification-as-RL Formulation (SOTA discussion)]] - rationale - report/drafts/state_of_the_art.md
- [[DQN (Deep Q-Network)]] - rationale - report/drafts/state_of_the_art.md
- [[DQN Fallback (when QRDQN load fails)]] - rationale - docs/Personal Research/models-parameters-and-validation-thesis-defense-report.md
- [[Data Structure and Canonical Schema Research Report]] - document - docs/Personal Research/data-structure-and-canonical-schema-research-report.md
- [[Deep Defense Research README]] - document - docs/Personal Research/deep-defense-research/README.md
- [[Deep Defense Datos, Esquema Canonico y Preprocesado]] - document - docs/Personal Research/deep-defense-research/02-datos-esquema-canonico-y-preprocesado.md
- [[Deep Defense Entorno RL, Algoritmo y Entrenamiento]] - document - docs/Personal Research/deep-defense-research/03-entorno-rl-algoritmo-y-entrenamiento.md
- [[Deep Defense Fundamentos y Objetivo]] - document - docs/Personal Research/deep-defense-research/01-fundamentos-y-objetivo.md
- [[Deep Defense Glosario y Preguntas de Tribunal]] - document - docs/Personal Research/deep-defense-research/06-glosario-y-preguntas-tribunal.md
- [[Deep Defense Phase 2 - Laboratorio, Inferencia y Riesgos]] - document - docs/Personal Research/deep-defense-research/05-phase2-laboratorio-inferencia-y-riesgos.md
- [[Deep Defense Validacion y Lectura de Resultados]] - document - docs/Personal Research/deep-defense-research/04-validacion-y-lectura-de-resultados.md
- [[Distributional RL  QRDQN Justification for NIDS]] - rationale - report/drafts/state_of_the_art.md
- [[Domain Shift Problem (Train vs Real Traffic)]] - rationale - docs/Personal Research/data-structure-and-canonical-schema-research-report.md
- [[External Validation via Lab-Captured Traffic]] - rationale - report/drafts/state_of_the_art.md
- [[Flow-Based Traffic Representation (CICFlowMeter)]] - rationale - report/drafts/state_of_the_art.md
- [[Hard Target Update (tau=1.0, periodic copy)]] - rationale - docs/Personal Research/qrdqn-research-report.md
- [[Lab Docker Traffic Generator]] - code - lab/docker/generator/gen_traffic.py
- [[Leave-One-Exact-CSV-Out Validation_1]] - rationale - docs/Personal Research/models-parameters-and-validation-thesis-defense-report.md
- [[Methodological Pitfalls in ML-Based NIDS]] - rationale - report/drafts/state_of_the_art.md
- [[Missingness Mask (76 binary indicators)]] - rationale - docs/Personal Research/data-structure-and-canonical-schema-research-report.md
- [[MlpPolicy net_arch=512,256 (main training)]] - rationale - docs/Personal Research/qrdqn-research-report.md
- [[Models, Parameters and Validation Thesis Defense Report]] - document - docs/Personal Research/models-parameters-and-validation-thesis-defense-report.md
- [[NSL-KDD Dataset (historical benchmark)]] - document - docs/Personal Research/deep-defense-research/02-datos-esquema-canonico-y-preprocesado.md
- [[Optuna Hyperparameter Search]] - rationale - docs/Personal Research/project-components-defense-research-report.md
- [[Percentile Clipping (p0.5p99.5 pre-scaling)]] - rationale - docs/Personal Research/data-structure-and-canonical-schema-research-report.md
- [[Phase 2 Offline Inference on Lab Traffic]] - rationale - docs/Personal Research/deep-defense-research/05-phase2-laboratorio-inferencia-y-riesgos.md
- [[Project Components Defense Research Report]] - document - docs/Personal Research/project-components-defense-research-report.md
- [[QRDQN Research Report]] - document - docs/Personal Research/qrdqn-research-report.md
- [[QRDQN exploration_fraction=0.005 (fast epsilon decay)]] - rationale - docs/Personal Research/qrdqn-research-report.md
- [[QRDQN n_quantiles=200 (implicit default)]] - rationale - docs/Personal Research/qrdqn-research-report.md
- [[QRDQN Distributional RL via Quantile Regression]] - rationale - docs/Personal Research/qrdqn-research-report.md
- [[Quantile Huber Loss]] - rationale - docs/Personal Research/qrdqn-research-report.md
- [[RF Day Split Result (accuracy=0.621, recall_attack=0.078)]] - document - runs/cicids2017/baseline_random_forest_comparison/results_rf.txt
- [[RF Leave-One-Out Wednesday Test (accuracy=0.637, recall_attack=0.0056)]] - document - runs/cicids2017/baseline_random_forest_comparison/results_rf.txt
- [[RF Random Split Result (accuracy=0.9988)]] - document - runs/cicids2017/baseline_random_forest_comparison/results_rf.txt
- [[RUN_ID Artifact System (reproducible runs)]] - rationale - docs/Personal Research/data-structure-and-canonical-schema-research-report.md
- [[Random Forest Baseline (n_estimators=200)]] - rationale - docs/Personal Research/models-parameters-and-validation-thesis-defense-report.md
- [[Random Forest Baseline Results (CICIDS2017)]] - document - runs/cicids2017/baseline_random_forest_comparison/results_rf.txt
- [[Reward Config Schema (tpfpfnomission)]] - rationale - tests/test_reward_config.py
- [[Reward Config Historical (fp=-2.0) vs Current (fp=-1.5) Discrepancy]] - rationale - docs/Personal Research/data-structure-and-canonical-schema-research-report.md
- [[Scaling Utilities (apply_percentile_clipping, apply_z_clipping)]] - code - src/scaling_utils.py
- [[StandardScaler applied to full 152-dim (mask scaled too)]] - rationale - docs/Personal Research/data-structure-and-canonical-schema-research-report.md
- [[StandardScaler fitted on train only (anti-leakage)]] - rationale - docs/Personal Research/data-structure-and-canonical-schema-research-report.md
- [[State of the Art Draft]] - document - report/drafts/state_of_the_art.md
- [[State of the Art TODO List]] - document - report/drafts/state_of_the_art_todo.md
- [[System Components Thesis Defense Research Report]] - document - docs/Personal Research/system-components-thesis-defense-report.md
- [[Z-Score Clipping (post-scaling)]] - rationale - docs/Personal Research/system-components-thesis-defense-report.md
- [[train_percentiles.npz artifact]] - rationale - docs/Personal Research/data-structure-and-canonical-schema-research-report.md

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/RL_Defender_Core_Design
SORT file.name ASC
```

## Connections to other communities
- 6 edges to [[_COMMUNITY_RL Training & Validation Patterns]]
- 1 edge to [[_COMMUNITY_Scaling & Clipping Utilities]]

## Top bridge nodes
- [[Data Structure and Canonical Schema Research Report]] - degree 14, connects to 1 community
- [[Deep Defense Entorno RL, Algoritmo y Entrenamiento]] - degree 6, connects to 1 community
- [[System Components Thesis Defense Research Report]] - degree 4, connects to 1 community
- [[Scaling Utilities (apply_percentile_clipping, apply_z_clipping)]] - degree 4, connects to 1 community
- [[Reward Config Schema (tpfpfnomission)]] - degree 2, connects to 1 community