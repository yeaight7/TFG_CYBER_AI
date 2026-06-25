# Graph Report - C:\Users\Rivero\Desktop\TFG_CYBER_AI  (2026-06-25)

## Corpus Check
- 89 files · ~165,569 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 265 nodes · 396 edges · 25 communities (16 shown, 9 thin omitted)
- Extraction: 84% EXTRACTED · 14% INFERRED · 0% AMBIGUOUS · INFERRED: 56 edges (avg confidence: 0.78)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Baseline Random Forest|Baseline Random Forest]]
- [[_COMMUNITY_Scaling & Clipping Utils|Scaling & Clipping Utils]]
- [[_COMMUNITY_RL Training Core|RL Training Core]]
- [[_COMMUNITY_Inference (Deprecated)|Inference (Deprecated)]]
- [[_COMMUNITY_Testing Splits|Testing Splits]]
- [[_COMMUNITY_Core Architecture Docs|Core Architecture Docs]]
- [[_COMMUNITY_RL Defender Environment|RL Defender Environment]]
- [[_COMMUNITY_Canonical Features & Schema|Canonical Features & Schema]]
- [[_COMMUNITY_Validation Checks|Validation Checks]]
- [[_COMMUNITY_Graphify Auto Update|Graphify Auto Update]]
- [[_COMMUNITY_Leave-One-Out Eval|Leave-One-Out Eval]]
- [[_COMMUNITY_Tensorboard Export|Tensorboard Export]]
- [[_COMMUNITY_Hparam Tuning|Hparam Tuning]]
- [[_COMMUNITY_RL Conceptual Design|RL Conceptual Design]]
- [[_COMMUNITY_Deep Research Report 1|Deep Research Report 1]]
- [[_COMMUNITY_Datasets & Risks|Datasets & Risks]]
- [[_COMMUNITY_Defense Presentations|Defense Presentations]]
- [[_COMMUNITY_Archive|Archive]]
- [[_COMMUNITY_Validation Report|Validation Report]]
- [[_COMMUNITY_Components Report|Components Report]]
- [[_COMMUNITY_QRDQN Report|QRDQN Report]]
- [[_COMMUNITY_System Defense Report|System Defense Report]]
- [[_COMMUNITY_CICIDS2017 Specs|CICIDS2017 Specs]]
- [[_COMMUNITY_NIDS Datasets|NIDS Datasets]]
- [[_COMMUNITY_Research 1|Research 1]]

## God Nodes (most connected - your core abstractions)
1. `RLDatasetDefenderEnv` - 15 edges
2. `load_cicids2017_split()` - 12 edges
3. `main()` - 11 edges
4. `decide_and_run()` - 10 edges
5. `main()` - 10 edges
6. `_prepare_cicids_features()` - 10 edges
7. `CICIDSLoadConfig` - 9 edges
8. `load_cicids2017_binary()` - 9 edges
9. `main()` - 9 edges
10. `map_to_canonical()` - 8 edges

## Surprising Connections (you probably didn't know these)
- `main()` --calls--> `map_to_canonical()`  [INFERRED]
  scripts/predict_real_traffic_v2.py → src/canonical_schema.py
- `main()` --calls--> `load_cicids2017_split()`  [INFERRED]
  scripts/verify_fixed_test_split.py → src/load_cicids2017.py
- `test_map_to_canonical_mask_logic()` --calls--> `map_to_canonical()`  [INFERRED]
  tests/test_canonical_schema.py → src/canonical_schema.py
- `test_sha256_of_array_stable()` --calls--> `_sha256_of_array()`  [INFERRED]
  tests/test_load_cicids2017.py → src/load_cicids2017.py
- `test_nested_prefix_indices_nested_and_stratified()` --calls--> `_stratified_nested_prefix_indices()`  [INFERRED]
  tests/test_load_cicids2017.py → src/load_cicids2017.py

## Hyperedges (group relationships)
- **Defense Preparation** — docs_DEFENSA_TFG_PROGRESO_md, docs_DEFENSA_TFG_SCRIPT_md, data_schema_research_report, validation_research_report, components_research_report [INFERRED 0.85]
- **Phase 2 Lab and Inference** — docs_AGENT_CONTEXT_md, docs_phase2_plan_md, docs_gcp_lab_md [INFERRED 0.90]
- **TFG_Positioning** — deep-research-report2, QRDQN, RLDatasetDefenderEnv, Cost_Sensitive_Reward [INFERRED 0.95]
- **Methodological_Critique** — deep-research-report1, deep-research-report2, Phase_2_Inference, CICIDS2017 [INFERRED 0.90]
- **Rigorous NIDS Evaluation Framework** — Temporal/Scenario Splits, External Validation (Lab Traffic), Cost-Sensitive IDS Reward, Methodological Risks in NIDS [INFERRED]
- **RL Agent Paradigm** — QRDQN Policy Agent, Dataset-as-Environment, Binary PERMIT/BLOCK Classification [INFERRED]
- **Thesis Methodology Pillars** — concept_canonical_feature_schema, concept_dataset_as_environment, concept_cost_sensitive_reward, concept_validation_ladder [INFERRED 0.85]
- **CICIDS2017 Evaluation Strategy** — concept_cicids2017_limitations, concept_validation_ladder, concept_evaluation_leakage [INFERRED 0.90]

## Communities (25 total, 9 thin omitted)

### Community 0 - "Baseline Random Forest"
Cohesion: 0.11
Nodes (32): evaluate_random_forest(), main(), Entrena un RandomForestClassifier sobre el dataset y devuelve el modelo., Evalúa el Random Forest y muestra matriz de confusión + informe de clasificación, train_random_forest(), CICIDSLoadConfig, _clean_rows(), _coerce_numeric_features() (+24 more)

### Community 1 - "Scaling & Clipping Utils"
Cohesion: 0.11
Nodes (19): apply_percentile_clipping(), apply_z_clipping(), scaling_utils.py — Clipping utilities for outlier handling in RL inference pipel, Clamp each feature to its training percentile range [p_low, p_high].      Appl, Clamp scaled features to [-max_z, +max_z].      Applied to scaled features **a, batched_predict(), compute_diagnostics(), compute_truth_metrics() (+11 more)

### Community 2 - "RL Training Core"
Cohesion: 0.17
Nodes (21): collect_environment_metadata(), configure_torch_runtime(), evaluate_model(), main(), make_env_fn(), _package_version(), parse_args(), train_rl_defender.py — Entrenamiento de agente defensor RL sobre CICIDS2017. (+13 more)

### Community 3 - "Inference (Deprecated)"
Cohesion: 0.12
Nodes (17): batched_predict(), load_model(), main(), maybe_convert_time_units(), CanonicalResult, get_canonical_feature_names(), get_observation_feature_names(), map_to_canonical() (+9 more)

### Community 4 - "Testing Splits"
Cohesion: 0.15
Nodes (18): fail(), main(), ok(), parse_args(), SHA-256 content hash of an ndarray, prefixed with dtype and shape so that     a, Deterministic, stratified, *nested* train subsample indices.      Per-class se, _sha256_of_array(), _stratified_nested_prefix_indices() (+10 more)

### Community 5 - "Core Architecture Docs"
Cohesion: 0.13
Nodes (15): Canonical Flow Schema, Phase 2 Offline Inference, MAIN_qrdqn_cicids2017_canonical_full_random_20260609_193655, Data Structure and Canonical Schema Research, Phase 2 Agent Context, Documentation Index, Audits Index, GCP Lab Setup (+7 more)

### Community 6 - "RL Defender Environment"
Cohesion: 0.15
Nodes (7): Calcula la recompensa en función de la etiqueta real, la acción         y la co, Entorno RL para un defensor que decide PERMIT/BLOCK sobre muestras etiquetadas., RLDatasetDefenderEnv, test_reward_logic_tp_fp_tn_fn(), test_unknown_label_reward(), test_env_initialization(), test_env_step_and_reset()

### Community 7 - "Canonical Features & Schema"
Cohesion: 0.21
Nodes (17): Canonical Feature Schema, CICIDS2017 Limitations, Cost-Sensitive Reward, Data Efficiency Experiments, Dataset as Environment, Distributional RL (QRDQN), Evaluation Leakage, Supervised Baselines (Random Forest) (+9 more)

### Community 8 - "Validation Checks"
Cohesion: 0.18
Nodes (12): BaseCallback, check_a_direct_eval(), check_b_shuffled_labels(), check_c_csv_split(), main(), parse_args(), ProgressCallback, validate_checks.py — Validación de resultados experimentales del agente RL.  I (+4 more)

### Community 9 - "Graphify Auto Update"
Cohesion: 0.34
Nodes (13): decide_and_run(), _diff_range(), _has_structural_patch(), _is_code(), _is_ignored(), _is_semantic_source(), main(), _normalize() (+5 more)

### Community 10 - "Leave-One-Out Eval"
Cohesion: 0.25
Nodes (13): _build_aggregate_results(), _compute_reward_total(), evaluate_model_direct(), main(), make_env_fn(), _metrics_from_confusion(), parse_args(), validate_leave_one_csv_out.py — Validación leave-one-exact-CSV-out para CICIDS20 (+5 more)

### Community 11 - "Tensorboard Export"
Cohesion: 0.46
Nodes (7): _find_event_dirs(), main(), parse_args(), _plot_scalar(), _read_scalars(), _safe_filename(), _update_artifact_manifest()

### Community 12 - "Hparam Tuning"
Cohesion: 0.36
Nodes (7): _evaluate_f1(), main(), objective(), parse_args(), tune_hparams.py -- Optimizacion de hiperparametros con Optuna para QRDQN sobre C, Evaluate model and return F1 score for attack class., Optuna objective: train QRDQN with suggested hparams, return F1 attack.

### Community 13 - "RL Conceptual Design"
Cohesion: 0.29
Nodes (8): Binary PERMIT/BLOCK Classification, Cost-Sensitive IDS Reward, Dataset-as-Environment, QRDQN Policy Agent, Research2.md, report-classification-dossier.md, report-qrdqn-deep-distributional-rl.md, report-reward-and-cost-sensitive-design-dossier.md

### Community 14 - "Deep Research Report 1"
Cohesion: 0.29
Nodes (7): Cost_Sensitive_Reward, Phase_2_Inference, QRDQN, RLDatasetDefenderEnv, deep-research-report1, deep-research-report2, state_of_the_art_todo

### Community 15 - "Datasets & Risks"
Cohesion: 0.29
Nodes (7): CICIDS2017 Dataset, External Validation (Lab Traffic), Methodological Risks in NIDS, Research3.md, Temporal/Scenario Splits, report-deep-dive.md, report-source-map.md

## Ambiguous Edges - Review These
- `Cost-Sensitive IDS Reward` → `Dataset-as-Environment`  [AMBIGUOUS]
   · relation: unknown

## Knowledge Gaps
- **54 isolated node(s):** `predict_real_traffic_v2.py — Robust Phase 2 offline inference pipeline.  Loads`, `If time columns look like seconds (median < 1), convert to microseconds.`, `Load QRDQN model, falling back to DQN if sb3_contrib is unavailable.`, `Run model.predict in batches to avoid OOM on large flow CSVs.`, `Compute z-score diagnostics on scaled features (first _N_CANON dims only).` (+49 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **9 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `Cost-Sensitive IDS Reward` and `Dataset-as-Environment`?**
  _Edge tagged AMBIGUOUS (relation: related to) - confidence is low._
- **Why does `load_cicids2017_split()` connect `Baseline Random Forest` to `Validation Checks`, `RL Training Core`, `Testing Splits`?**
  _High betweenness centrality (0.208) - this node is a cross-community bridge._
- **Why does `map_to_canonical()` connect `Inference (Deprecated)` to `Baseline Random Forest`, `Scaling & Clipping Utils`?**
  _High betweenness centrality (0.184) - this node is a cross-community bridge._
- **Why does `_prepare_cicids_features()` connect `Baseline Random Forest` to `Inference (Deprecated)`?**
  _High betweenness centrality (0.175) - this node is a cross-community bridge._
- **Are the 6 inferred relationships involving `RLDatasetDefenderEnv` (e.g. with `ProgressCallback` and `_evaluate_f1()`) actually correct?**
  _`RLDatasetDefenderEnv` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `load_cicids2017_split()` (e.g. with `main()` and `main()`) actually correct?**
  _`load_cicids2017_split()` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `main()` (e.g. with `map_to_canonical()` and `apply_percentile_clipping()`) actually correct?**
  _`main()` has 3 INFERRED edges - model-reasoned connections that need verification._