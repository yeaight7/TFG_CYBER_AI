# Graph Report - .  (2026-05-15)

## Corpus Check
- 53 files · ~91,945 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 294 nodes · 408 edges · 28 communities (12 shown, 16 thin omitted)
- Extraction: 87% EXTRACTED · 13% INFERRED · 0% AMBIGUOUS · INFERRED: 54 edges (avg confidence: 0.83)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Thesis Research and Experiments|Thesis Research and Experiments]]
- [[_COMMUNITY_RF Baseline Pipeline|RF Baseline Pipeline]]
- [[_COMMUNITY_CICIDS2017 and Flow Features|CICIDS2017 and Flow Features]]
- [[_COMMUNITY_RL Defender Environment|RL Defender Environment]]
- [[_COMMUNITY_Distributional RL Agent|Distributional RL Agent]]
- [[_COMMUNITY_Inference and Scaling Utils|Inference and Scaling Utils]]
- [[_COMMUNITY_Real Traffic Prediction|Real Traffic Prediction]]
- [[_COMMUNITY_Graph Auto-Update System|Graph Auto-Update System]]
- [[_COMMUNITY_Validation and Sanity Checks|Validation and Sanity Checks]]
- [[_COMMUNITY_Leave-One-CSV-Out Validation|Leave-One-CSV-Out Validation]]
- [[_COMMUNITY_Hyperparameter Tuning|Hyperparameter Tuning]]
- [[_COMMUNITY_KDD Datasets|KDD Datasets]]
- [[_COMMUNITY_Project Invariants|Project Invariants]]
- [[_COMMUNITY_Cross-CSV Validation|Cross-CSV Validation]]
- [[_COMMUNITY_Best Experiment Run|Best Experiment Run]]
- [[_COMMUNITY_Binary PERMITBLOCK Decision|Binary PERMIT/BLOCK Decision]]
- [[_COMMUNITY_PyTorch Framework|PyTorch Framework]]
- [[_COMMUNITY_Optuna Tuning|Optuna Tuning]]
- [[_COMMUNITY_GCP Lab Safety|GCP Lab Safety]]
- [[_COMMUNITY_Documentation Index|Documentation Index]]
- [[_COMMUNITY_Experiment Archive|Experiment Archive]]
- [[_COMMUNITY_Fair RL Classification|Fair RL Classification]]
- [[_COMMUNITY_IDS2018 Dataset|IDS2018 Dataset]]
- [[_COMMUNITY_Bot-IoT Dataset|Bot-IoT Dataset]]
- [[_COMMUNITY_TON-IoT Dataset|TON-IoT Dataset]]
- [[_COMMUNITY_UGR16 Dataset|UGR16 Dataset]]
- [[_COMMUNITY_Flow-Based NIDS Survey|Flow-Based NIDS Survey]]

## God Nodes (most connected - your core abstractions)
1. `RLDatasetDefenderEnv` - 16 edges
2. `CICIDS2017 Dataset` - 12 edges
3. `decide_and_run()` - 10 edges
4. `main()` - 10 edges
5. `_prepare_cicids_features()` - 10 edges
6. `Research Source Matrix` - 10 edges
7. `CICIDSLoadConfig` - 9 edges
8. `load_cicids2017_binary()` - 9 edges
9. `main()` - 9 edges
10. `map_to_canonical()` - 8 edges

## Surprising Connections (you probably didn't know these)
- `Anti-Leakage Policy` --semantically_similar_to--> `Arp2020DosDontsMLSecurity`  [INFERRED] [semantically similar]
  AGENTS.md → Research/CITATION_PLAN.md
- `Claim: Cross-Domain Generalization is Known Weakness` --semantically_similar_to--> `Domain Shift Risk (Phase 2)`  [INFERRED] [semantically similar]
  Research/CLAIMS_BANK.md → docs/AGENT_CONTEXT.md
- `main()` --calls--> `map_to_canonical()`  [INFERRED]
  scripts/predict_real_traffic_v2.py → src/canonical_schema.py
- `test_map_to_canonical_mask_logic()` --calls--> `map_to_canonical()`  [INFERRED]
  tests/test_canonical_schema.py → src/canonical_schema.py
- `test_reward_logic_tp_fp_tn_fn()` --calls--> `RLDatasetDefenderEnv`  [INFERRED]
  tests/test_reward_config.py → src/rl_defender_env.py

## Hyperedges (group relationships)
- **Canonical Observation Pipeline (schema + mask + 152-dim vector)** — readme_canonical_schema, readme_missingness_mask, readme_observation_vector_152 [EXTRACTED 1.00]
- **RL Defender Training Stack (Gymnasium + SB3 + QRDQN)** — requirements_gymnasium, requirements_stable_baselines3, requirements_sb3contrib, readme_qrdqn_algorithm [EXTRACTED 1.00]
- **Phase 2 Inference Chain (CICFlowMeter + canonical mapping + predict_v2)** — gcp_lab_cicflowmeter, readme_canonical_schema, readme_predict_real_traffic_v2, agent_context_domain_shift_risk [EXTRACTED 0.95]
- **QRDQN + CICIDS2017 + PERMIT/BLOCK = Core TFG Experimental Pipeline** — deep_research_report2_qrdqn_agent, deep_dive_cicids2017, deep_research_report2_permit_block_formulation, experimental_board_152obs_gymnasium, reward_dossier_cost_sensitive_reward [EXTRACTED 1.00]
- **Leakage Controls + Strict Splits + External Validation = Evaluation Rigor Strategy** — methodology_handoff_leakage_controls, experimental_board_branch_e_strict_splits, methodology_handoff_external_validation_plans, research_index_evaluation_methodology_area [INFERRED 0.95]
- **Distributional RL (QRDQN) + Cost-Sensitive Reward + FN/FP Asymmetry = Risk-Aware Defender Design** — qrdqn_dossier_risk_aware_policy, reward_dossier_cost_sensitive_reward, reward_dossier_fn_fp_asymmetry [INFERRED 0.85]

## Communities (28 total, 16 thin omitted)

### Community 0 - "Thesis Research and Experiments"
Cohesion: 0.05
Nodes (59): Domain Shift Risk (Phase 2), Phase 2 Scope and Guardrails, Anti-Leakage Policy, C03_qrdqn_cicids2017_canonical_full_random_20260223_232439, CICIDS2017 QRDQN Experiment History, Arp2020DosDontsMLSecurity, Bellemare2017DistributionalRL, Dabney2018QRDQN (Distributional RL with Quantile Regression) (+51 more)

### Community 1 - "RF Baseline Pipeline"
Cohesion: 0.11
Nodes (32): evaluate_random_forest(), main(), Entrena un RandomForestClassifier sobre el dataset y devuelve el modelo., Evalúa el Random Forest y muestra matriz de confusión + informe de clasificación, train_random_forest(), CICIDSLoadConfig, _clean_rows(), _coerce_numeric_features() (+24 more)

### Community 2 - "CICIDS2017 and Flow Features"
Cohesion: 0.09
Nodes (32): B-Profile System for Benign Traffic Generation, 76 Canonical Flow Features Schema, CICFlowMeter Flow Feature Extraction Tool, CICIDS2017 Dataset, ENGELEN2021 (Troubleshooting CICIDS2017, WTMC), LANVIN2023 (Errors in CICIDS2017 Dataset), SHARAFALDIN2018 (Toward Generating a New IDS Dataset), APRUZZESE2022 (Cross-Evaluation of ML-based NIDS) (+24 more)

### Community 3 - "RL Defender Environment"
Cohesion: 0.11
Nodes (14): Calcula la recompensa en función de la etiqueta real, la acción         y la co, Entorno RL para un defensor que decide PERMIT/BLOCK sobre muestras etiquetadas., RLDatasetDefenderEnv, evaluate_model(), main(), make_env_fn(), parse_args(), train_rl_defender.py — Entrenamiento de agente defensor RL sobre CICIDS2017. (+6 more)

### Community 4 - "Distributional RL Agent"
Cohesion: 0.09
Nodes (25): Classification-as-RL Formulation, Dataset-as-Environment Pattern, DQLIDS2021 (Deep Q-Learning based RL for NIDS, NSL-KDD), DQNIDS2026 (DQN-IDS: Open-Set RL NIDS, NDSS), RLCC2023 (Reinforcement Learning Cost-Sensitive Classifier), Offline Flow-Level PERMIT/BLOCK RL Formulation, QRDQN Agent for PERMIT/BLOCK, BELLEMARE2017 (Distributional RL, C51) (+17 more)

### Community 5 - "Inference and Scaling Utils"
Cohesion: 0.11
Nodes (19): apply_percentile_clipping(), apply_z_clipping(), scaling_utils.py — Clipping utilities for outlier handling in RL inference pipel, Clamp each feature to its training percentile range [p_low, p_high].      Appl, Clamp scaled features to [-max_z, +max_z].      Applied to scaled features **a, batched_predict(), compute_diagnostics(), compute_truth_metrics() (+11 more)

### Community 6 - "Real Traffic Prediction"
Cohesion: 0.12
Nodes (17): batched_predict(), load_model(), main(), maybe_convert_time_units(), CanonicalResult, get_canonical_feature_names(), get_observation_feature_names(), map_to_canonical() (+9 more)

### Community 7 - "Graph Auto-Update System"
Cohesion: 0.24
Nodes (14): decide_and_run(), _diff_range(), _has_structural_patch(), _is_code(), _is_ignored(), _is_semantic_source(), main(), _normalize() (+6 more)

### Community 8 - "Validation and Sanity Checks"
Cohesion: 0.18
Nodes (12): BaseCallback, check_a_direct_eval(), check_b_shuffled_labels(), check_c_csv_split(), main(), parse_args(), ProgressCallback, validate_checks.py — Validación de resultados experimentales del agente RL.  I (+4 more)

### Community 9 - "Leave-One-CSV-Out Validation"
Cohesion: 0.25
Nodes (13): _build_aggregate_results(), _compute_reward_total(), evaluate_model_direct(), main(), make_env_fn(), _metrics_from_confusion(), parse_args(), validate_leave_one_csv_out.py — Validación leave-one-exact-CSV-out para CICIDS20 (+5 more)

### Community 10 - "Hyperparameter Tuning"
Cohesion: 0.36
Nodes (7): _evaluate_f1(), main(), objective(), parse_args(), tune_hparams.py -- Optimizacion de hiperparametros con Optuna para QRDQN sobre C, Evaluate model and return F1 score for attack class., Optuna objective: train QRDQN with suggested hparams, return F1 attack.

## Knowledge Gaps
- **93 isolated node(s):** `predict_real_traffic_v2.py — Robust Phase 2 offline inference pipeline.  Loads`, `If time columns look like seconds (median < 1), convert to microseconds.`, `Load QRDQN model, falling back to DQN if sb3_contrib is unavailable.`, `Run model.predict in batches to avoid OOM on large flow CSVs.`, `Compute z-score diagnostics on scaled features (first _N_CANON dims only).` (+88 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **16 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `map_to_canonical()` connect `Real Traffic Prediction` to `RF Baseline Pipeline`, `Inference and Scaling Utils`?**
  _High betweenness centrality (0.114) - this node is a cross-community bridge._
- **Why does `_prepare_cicids_features()` connect `RF Baseline Pipeline` to `Real Traffic Prediction`?**
  _High betweenness centrality (0.104) - this node is a cross-community bridge._
- **Why does `load_cicids2017_split()` connect `RF Baseline Pipeline` to `Validation and Sanity Checks`, `RL Defender Environment`?**
  _High betweenness centrality (0.081) - this node is a cross-community bridge._
- **Are the 7 inferred relationships involving `RLDatasetDefenderEnv` (e.g. with `ProgressCallback` and `evaluate_model()`) actually correct?**
  _`RLDatasetDefenderEnv` has 7 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `main()` (e.g. with `map_to_canonical()` and `apply_percentile_clipping()`) actually correct?**
  _`main()` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `_prepare_cicids_features()` (e.g. with `map_to_canonical()` and `test_prepare_cicids_features_binary_labels()`) actually correct?**
  _`_prepare_cicids_features()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `predict_real_traffic_v2.py — Robust Phase 2 offline inference pipeline.  Loads`, `If time columns look like seconds (median < 1), convert to microseconds.`, `Load QRDQN model, falling back to DQN if sb3_contrib is unavailable.` to the rest of the system?**
  _93 weakly-connected nodes found - possible documentation gaps or missing edges._