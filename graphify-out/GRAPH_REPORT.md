# Graph Report - .  (2026-05-15)

## Corpus Check
- 67 files · ~109,920 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 413 nodes · 596 edges · 32 communities (14 shown, 18 thin omitted)
- Extraction: 88% EXTRACTED · 12% INFERRED · 0% AMBIGUOUS · INFERRED: 70 edges (avg confidence: 0.83)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_RL Defender Core Design|RL Defender Core Design]]
- [[_COMMUNITY_CICIDS2017 Research & Experiments|CICIDS2017 Research & Experiments]]
- [[_COMMUNITY_RF Baseline Module|RF Baseline Module]]
- [[_COMMUNITY_CICIDS2017 Dataset & Citations|CICIDS2017 Dataset & Citations]]
- [[_COMMUNITY_RL Training & Validation Patterns|RL Training & Validation Patterns]]
- [[_COMMUNITY_RL Environment (Code)|RL Environment (Code)]]
- [[_COMMUNITY_Scaling & Clipping Utilities|Scaling & Clipping Utilities]]
- [[_COMMUNITY_RL for NIDS Literature|RL for NIDS Literature]]
- [[_COMMUNITY_Graphify Auto-Update|Graphify Auto-Update]]
- [[_COMMUNITY_Deprecated Inference v1|Deprecated Inference v1]]
- [[_COMMUNITY_Canonical Feature Schema|Canonical Feature Schema]]
- [[_COMMUNITY_Validation Checks Module|Validation Checks Module]]
- [[_COMMUNITY_Leave-One-Out Validation|Leave-One-Out Validation]]
- [[_COMMUNITY_Hyperparameter Tuning|Hyperparameter Tuning]]
- [[_COMMUNITY_NSL-KDD Dataset|NSL-KDD Dataset]]
- [[_COMMUNITY_Project Invariants|Project Invariants]]
- [[_COMMUNITY_CSV-Out Validation Strategy|CSV-Out Validation Strategy]]
- [[_COMMUNITY_Best Run Artifact (C03)|Best Run Artifact (C03)]]
- [[_COMMUNITY_PERMITBLOCK Decision|PERMIT/BLOCK Decision]]
- [[_COMMUNITY_PyTorch Dependency|PyTorch Dependency]]
- [[_COMMUNITY_Optuna Dependency|Optuna Dependency]]
- [[_COMMUNITY_Lab Safety Guardrails|Lab Safety Guardrails]]
- [[_COMMUNITY_Documentation Index|Documentation Index]]
- [[_COMMUNITY_Experiment Archive|Experiment Archive]]
- [[_COMMUNITY_Fair RL Classification|Fair RL Classification]]
- [[_COMMUNITY_CIC-IDS2018 Dataset|CIC-IDS2018 Dataset]]
- [[_COMMUNITY_Bot-IoT Dataset|Bot-IoT Dataset]]
- [[_COMMUNITY_TON_IoT Dataset|TON_IoT Dataset]]
- [[_COMMUNITY_UGR16 Network Dataset|UGR16 Network Dataset]]
- [[_COMMUNITY_Flow-Based NIDS Survey|Flow-Based NIDS Survey]]
- [[_COMMUNITY_CICIDS Load Config|CICIDS Load Config]]
- [[_COMMUNITY_NSL-KDD Column Schema|NSL-KDD Column Schema]]

## God Nodes (most connected - your core abstractions)
1. `RLDatasetDefenderEnv gymnasium environment` - 19 edges
2. `RLDatasetDefenderEnv` - 16 edges
3. `Data Structure and Canonical Schema Research Report` - 14 edges
4. `decide_and_run()` - 13 edges
5. `CICIDS2017 Dataset` - 12 edges
6. `State of the Art Draft` - 12 edges
7. `main()` - 10 edges
8. `_prepare_cicids_features()` - 10 edges
9. `Research Source Matrix` - 10 edges
10. `QRDQN Research Report` - 10 edges

## Surprising Connections (you probably didn't know these)
- `Classification-as-RL Formulation (SOTA discussion)` --semantically_similar_to--> `Classification-as-RL / Contextual Bandit Equivalence`  [INFERRED] [semantically similar]
  report/drafts/state_of_the_art.md → docs/Personal Research/data-structure-and-canonical-schema-research-report.md
- `Anti-Leakage Policy` --semantically_similar_to--> `Arp2020DosDontsMLSecurity`  [INFERRED] [semantically similar]
  AGENTS.md → Research/CITATION_PLAN.md
- `Claim: Cross-Domain Generalization is Known Weakness` --semantically_similar_to--> `Domain Shift Risk (Phase 2)`  [INFERRED] [semantically similar]
  Research/CLAIMS_BANK.md → docs/AGENT_CONTEXT.md
- `CICIDS2017_TO_CANON mapping dict` --semantically_similar_to--> `FLOWMETER_PY_TO_CANON mapping in v2`  [INFERRED] [semantically similar]
  src/canonical_schema.py → scripts/predict_real_traffic_v2.py
- `Data Structure and Canonical Schema Research Report` --references--> `RLDatasetDefenderEnv gymnasium environment`  [EXTRACTED]
  docs/Personal Research/data-structure-and-canonical-schema-research-report.md → src/rl_defender_env.py

## Hyperedges (group relationships)
- **Canonical Observation Pipeline (schema + mask + 152-dim vector)** — readme_canonical_schema, readme_missingness_mask, readme_observation_vector_152 [EXTRACTED 1.00]
- **RL Defender Training Stack (Gymnasium + SB3 + QRDQN)** — requirements_gymnasium, requirements_stable_baselines3, requirements_sb3contrib, readme_qrdqn_algorithm [EXTRACTED 1.00]
- **Phase 2 Inference Chain (CICFlowMeter + canonical mapping + predict_v2)** — gcp_lab_cicflowmeter, readme_canonical_schema, readme_predict_real_traffic_v2, agent_context_domain_shift_risk [EXTRACTED 0.95]
- **QRDQN + CICIDS2017 + PERMIT/BLOCK = Core TFG Experimental Pipeline** — deep_research_report2_qrdqn_agent, deep_dive_cicids2017, deep_research_report2_permit_block_formulation, experimental_board_152obs_gymnasium, reward_dossier_cost_sensitive_reward [EXTRACTED 1.00]
- **Leakage Controls + Strict Splits + External Validation = Evaluation Rigor Strategy** — methodology_handoff_leakage_controls, experimental_board_branch_e_strict_splits, methodology_handoff_external_validation_plans, research_index_evaluation_methodology_area [INFERRED 0.95]
- **Distributional RL (QRDQN) + Cost-Sensitive Reward + FN/FP Asymmetry = Risk-Aware Defender Design** — qrdqn_dossier_risk_aware_policy, reward_dossier_cost_sensitive_reward, reward_dossier_fn_fp_asymmetry [INFERRED 0.85]
- **Canonical Schema Feature Pipeline (map + mask + scale)** — canonical_schema_map_to_canonical, canonical_schema_missingness_mask, load_cicids2017__prepare_cicids_features, load_nsl_kdd_load_nsl_kdd_binary, rl_defender_env_RLDatasetDefenderEnv [INFERRED 0.95]
- **Phase 2 Real Traffic Inference Pipeline (v2)** — predict_v2_main, scaling_utils_apply_percentile_clipping, scaling_utils_apply_z_clipping, predict_v2_compute_diagnostics, canonical_schema_map_to_canonical [EXTRACTED 1.00]
- **RL Defender Training Loop (env + QRDQN + CICIDS2017)** — train_rl_defender_main, rl_defender_env_RLDatasetDefenderEnv, load_cicids2017_load_cicids2017_split, train_rl_defender_QRDQN [EXTRACTED 1.00]
- **Multi-Stage Validation Ladder (A/B/C + Leave-One-Out)** — check_a_direct_eval, check_b_shuffled_labels, check_c_csv_split, leave_one_csv_out_validation [EXTRACTED 1.00]
- **Robust Phase 2 Inference Pipeline** — percentile_clipping, zscore_clipping, standardscaler_fit_train_only, domain_shift_problem [EXTRACTED 1.00]
- **Reproducible Run Artifact System** — run_id_artifact_system, train_percentiles_npz, reward_config_historical_vs_current, best_run_C03 [INFERRED 0.85]

## Communities (32 total, 18 thin omitted)

### Community 0 - "RL Defender Core Design"
Cohesion: 0.06
Nodes (60): Anti-Leakage Policy (drop IPs, timestamps, Flow IDs, ports), Asymmetric Cost Reward Design (FN >> FP penalty), Best Run C03 (accuracy=0.99859, random split), C51 (Categorical DQN, distributional RL predecessor), 152-Dimensional Observation Vector, Canonical Schema (76 Flow Features), Check A: Direct Evaluation (anti-env-dependency), Check B: Shuffled Labels Anti-Leakage Test (+52 more)

### Community 1 - "CICIDS2017 Research & Experiments"
Cohesion: 0.05
Nodes (59): Domain Shift Risk (Phase 2), Phase 2 Scope and Guardrails, Anti-Leakage Policy, C03_qrdqn_cicids2017_canonical_full_random_20260223_232439, CICIDS2017 QRDQN Experiment History, Arp2020DosDontsMLSecurity, Bellemare2017DistributionalRL, Dabney2018QRDQN (Distributional RL with Quantile Regression) (+51 more)

### Community 2 - "RF Baseline Module"
Cohesion: 0.11
Nodes (32): evaluate_random_forest(), main(), Entrena un RandomForestClassifier sobre el dataset y devuelve el modelo., Evalúa el Random Forest y muestra matriz de confusión + informe de clasificación, train_random_forest(), CICIDSLoadConfig, _clean_rows(), _coerce_numeric_features() (+24 more)

### Community 3 - "CICIDS2017 Dataset & Citations"
Cohesion: 0.09
Nodes (34): B-Profile System for Benign Traffic Generation, 76 Canonical Flow Features Schema, CICFlowMeter Flow Feature Extraction Tool, CICIDS2017 Dataset, ENGELEN2021 (Troubleshooting CICIDS2017, WTMC), LANVIN2023 (Errors in CICIDS2017 Dataset), SHARAFALDIN2018 (Toward Generating a New IDS Dataset), APRUZZESE2022 (Cross-Evaluation of ML-based NIDS) (+26 more)

### Community 4 - "RL Training & Validation Patterns"
Cohesion: 0.08
Nodes (31): evaluate_random_forest function, baseline_random_forest main (3-sweep eval), train_random_forest function, Classification-as-RL / Contextual Bandit Equivalence, DummyVecEnv + Monitor Wrappers, Episode Mechanics: train shuffle / test deterministic, list_cicids2017_csv_files function, load_cicids2017_binary function (+23 more)

### Community 5 - "RL Environment (Code)"
Cohesion: 0.11
Nodes (14): Calcula la recompensa en función de la etiqueta real, la acción         y la co, Entorno RL para un defensor que decide PERMIT/BLOCK sobre muestras etiquetadas., RLDatasetDefenderEnv, evaluate_model(), main(), make_env_fn(), parse_args(), train_rl_defender.py — Entrenamiento de agente defensor RL sobre CICIDS2017. (+6 more)

### Community 6 - "Scaling & Clipping Utilities"
Cohesion: 0.11
Nodes (20): apply_percentile_clipping(), apply_z_clipping(), Distribution Shift Defense via Dual Clipping, scaling_utils.py — Clipping utilities for outlier handling in RL inference pipel, Clamp each feature to its training percentile range [p_low, p_high].      Appl, Clamp scaled features to [-max_z, +max_z].      Applied to scaled features **a, batched_predict(), compute_diagnostics() (+12 more)

### Community 7 - "RL for NIDS Literature"
Cohesion: 0.09
Nodes (23): Classification-as-RL Formulation, Dataset-as-Environment Pattern, DQLIDS2021 (Deep Q-Learning based RL for NIDS, NSL-KDD), DQNIDS2026 (DQN-IDS: Open-Set RL NIDS, NDSS), RLCC2023 (Reinforcement Learning Cost-Sensitive Classifier), BELLEMARE2017 (Distributional RL, C51), DABNEY2017 (QR-DQN, Distributional RL with Quantile Regression, AAAI), DQN Foundation (Deep Q-Network) (+15 more)

### Community 8 - "Graphify Auto-Update"
Cohesion: 0.17
Nodes (18): _has_structural_patch diff analyzer, _is_semantic_source path classifier, _rebuild_code_graph caller, decide_and_run(), _diff_range(), _has_structural_patch(), _is_code(), _is_ignored() (+10 more)

### Community 9 - "Deprecated Inference v1"
Cohesion: 0.12
Nodes (17): batched_predict(), load_model(), main(), maybe_convert_time_units(), CanonicalResult, get_canonical_feature_names(), get_observation_feature_names(), map_to_canonical() (+9 more)

### Community 10 - "Canonical Feature Schema"
Cohesion: 0.11
Nodes (22): CICIDS2017_TO_CANON mapping dict, CanonicalResult dataclass, FEATURES_CANON (76 canonical network flow features), NSL_KDD_TO_CANON mapping dict (partial), map_to_canonical function, Missingness Mask Design (observation = features + mask), _prepare_cicids_features internal function, load_nsl_kdd_binary function (+14 more)

### Community 11 - "Validation Checks Module"
Cohesion: 0.18
Nodes (12): BaseCallback, check_a_direct_eval(), check_b_shuffled_labels(), check_c_csv_split(), main(), parse_args(), ProgressCallback, validate_checks.py — Validación de resultados experimentales del agente RL.  I (+4 more)

### Community 12 - "Leave-One-Out Validation"
Cohesion: 0.25
Nodes (13): _build_aggregate_results(), _compute_reward_total(), evaluate_model_direct(), main(), make_env_fn(), _metrics_from_confusion(), parse_args(), validate_leave_one_csv_out.py — Validación leave-one-exact-CSV-out para CICIDS20 (+5 more)

### Community 13 - "Hyperparameter Tuning"
Cohesion: 0.36
Nodes (7): _evaluate_f1(), main(), objective(), parse_args(), tune_hparams.py -- Optimizacion de hiperparametros con Optuna para QRDQN sobre C, Evaluate model and return F1 score for attack class., Optuna objective: train QRDQN with suggested hparams, return F1 attack.

## Knowledge Gaps
- **125 isolated node(s):** `predict_real_traffic_v2.py — Robust Phase 2 offline inference pipeline.  Loads`, `If time columns look like seconds (median < 1), convert to microseconds.`, `Load QRDQN model, falling back to DQN if sb3_contrib is unavailable.`, `Run model.predict in batches to avoid OOM on large flow CSVs.`, `Compute z-score diagnostics on scaled features (first _N_CANON dims only).` (+120 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **18 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `map_to_canonical()` connect `Deprecated Inference v1` to `RF Baseline Module`, `Scaling & Clipping Utilities`?**
  _High betweenness centrality (0.217) - this node is a cross-community bridge._
- **Why does `main()` connect `Scaling & Clipping Utilities` to `Deprecated Inference v1`?**
  _High betweenness centrality (0.205) - this node is a cross-community bridge._
- **Why does `_prepare_cicids_features()` connect `RF Baseline Module` to `Deprecated Inference v1`?**
  _High betweenness centrality (0.182) - this node is a cross-community bridge._
- **Are the 7 inferred relationships involving `RLDatasetDefenderEnv` (e.g. with `ProgressCallback` and `evaluate_model()`) actually correct?**
  _`RLDatasetDefenderEnv` has 7 INFERRED edges - model-reasoned connections that need verification._
- **What connects `predict_real_traffic_v2.py — Robust Phase 2 offline inference pipeline.  Loads`, `If time columns look like seconds (median < 1), convert to microseconds.`, `Load QRDQN model, falling back to DQN if sb3_contrib is unavailable.` to the rest of the system?**
  _125 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `RL Defender Core Design` be split into smaller, more focused modules?**
  _Cohesion score 0.06 - nodes in this community are weakly interconnected._
- **Should `CICIDS2017 Research & Experiments` be split into smaller, more focused modules?**
  _Cohesion score 0.05 - nodes in this community are weakly interconnected._