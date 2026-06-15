# Graph Report - .  (2026-06-15)

## Corpus Check
- 91 files · ~165,412 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 305 nodes · 456 edges · 32 communities (28 shown, 4 thin omitted)
- Extraction: 87% EXTRACTED · 13% INFERRED · 0% AMBIGUOUS · INFERRED: 59 edges (avg confidence: 0.81)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Project Overview|Project Overview]]
- [[_COMMUNITY_Test Split Validation|Test Split Validation]]
- [[_COMMUNITY_CICIDS2017 Preprocessing|CICIDS2017 Preprocessing]]
- [[_COMMUNITY_Inference Diagnostics|Inference Diagnostics]]
- [[_COMMUNITY_RL Defender Training|RL Defender Training]]
- [[_COMMUNITY_Canonical Schema Setup|Canonical Schema Setup]]
- [[_COMMUNITY_Graphify Auto Update|Graphify Auto Update]]
- [[_COMMUNITY_RL Environment Config|RL Environment Config]]
- [[_COMMUNITY_Data Load Tools|Data Load Tools]]
- [[_COMMUNITY_Leave One CSV Out|Leave One CSV Out]]
- [[_COMMUNITY_Hyperparams & Rewards|Hyperparams & Rewards]]
- [[_COMMUNITY_Tensorboard Exports|Tensorboard Exports]]
- [[_COMMUNITY_Hyperparam Tuning|Hyperparam Tuning]]
- [[_COMMUNITY_State of Art Research|State of Art Research]]
- [[_COMMUNITY_Experimental Boards|Experimental Boards]]
- [[_COMMUNITY_QRDQN Methods|QRDQN Methods]]
- [[_COMMUNITY_Features Map|Features Map]]
- [[_COMMUNITY_NSLKDD Experiments|NSLKDD Experiments]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 31|Community 31]]

## God Nodes (most connected - your core abstractions)
1. `RLDatasetDefenderEnv` - 15 edges
2. `decide_and_run()` - 13 edges
3. `load_cicids2017_split()` - 12 edges
4. `main()` - 11 edges
5. `main()` - 10 edges
6. `_prepare_cicids_features()` - 10 edges
7. `CICIDSLoadConfig` - 9 edges
8. `load_cicids2017_binary()` - 9 edges
9. `main()` - 9 edges
10. `predict_real_traffic_v2 main pipeline` - 9 edges

## Surprising Connections (you probably didn't know these)
- `CICIDS2017_TO_CANON mapping dict` --semantically_similar_to--> `FLOWMETER_PY_TO_CANON mapping in v2`  [INFERRED] [semantically similar]
  src/canonical_schema.py → scripts/predict_real_traffic_v2.py
- `main()` --calls--> `map_to_canonical()`  [INFERRED]
  scripts/predict_real_traffic_v2.py → src/canonical_schema.py
- `test_map_to_canonical_mask_logic()` --calls--> `map_to_canonical()`  [INFERRED]
  tests/test_canonical_schema.py → src/canonical_schema.py
- `test_sha256_of_array_stable()` --calls--> `_sha256_of_array()`  [INFERRED]
  tests/test_load_cicids2017.py → src/load_cicids2017.py
- `test_nested_prefix_indices_nested_and_stratified()` --calls--> `_stratified_nested_prefix_indices()`  [INFERRED]
  tests/test_load_cicids2017.py → src/load_cicids2017.py

## Hyperedges (group relationships)
- **Phase 2 Real Traffic Inference Pipeline (v2)** — predict_v2_main, scaling_utils_apply_percentile_clipping, scaling_utils_apply_z_clipping, predict_v2_compute_diagnostics, canonical_schema_map_to_canonical [EXTRACTED 1.00]
- **Canonical Schema Feature Pipeline (map + mask + scale)** — canonical_schema_map_to_canonical, canonical_schema_missingness_mask, load_cicids2017__prepare_cicids_features, load_nsl_kdd_load_nsl_kdd_binary, rl_defender_env_RLDatasetDefenderEnv [INFERRED 0.95]
- **Validation Suite** — validation_check_a, validation_check_b, validation_check_c, leave_one_exact_csv_out [INFERRED 0.90]
- **TFG Core Logic Scripts** — load_cicids2017, train_rl_defender, validate_checks, validate_leave_one_csv_out, baseline_random_forest [INFERRED 0.80]

## Communities (32 total, 4 thin omitted)

### Community 0 - "Project Overview"
Cohesion: 0.09
Nodes (22): Anti-Leakage Policy (drop IPs, timestamps, Flow IDs, ports), CICIDS2017 Dataset, CICIDS2017 Dataset (8 CSVs, 5-day capture), Deep Defense: Fundamentos y Objetivo, Deep Defense: Datos, Esquema Canonico y Preprocesado, Deep Defense: Phase 2 - Laboratorio, Inferencia y Riesgos, Deep Defense: Glosario y Preguntas de Tribunal, Domain Shift (+14 more)

### Community 1 - "Test Split Validation"
Cohesion: 0.08
Nodes (32): BaseCallback, fail(), main(), ok(), parse_args(), load_cicids2017_split(), SHA-256 content hash of an ndarray, prefixed with dtype and shape so that     a, Deterministic, stratified, *nested* train subsample indices.      Per-class se (+24 more)

### Community 2 - "CICIDS2017 Preprocessing"
Cohesion: 0.12
Nodes (30): evaluate_random_forest(), main(), Entrena un RandomForestClassifier sobre el dataset y devuelve el modelo., Evalúa el Random Forest y muestra matriz de confusión + informe de clasificación, train_random_forest(), CICIDSLoadConfig, _clean_rows(), _coerce_numeric_features() (+22 more)

### Community 3 - "Inference Diagnostics"
Cohesion: 0.09
Nodes (24): NSL-KDD Dataset (historical benchmark), Optuna Hyperparameter Search, Project Components Defense Research Report, Scaling Utilities (apply_percentile_clipping, apply_z_clipping), apply_percentile_clipping(), apply_z_clipping(), Distribution Shift Defense via Dual Clipping, scaling_utils.py — Clipping utilities for outlier handling in RL inference pipel (+16 more)

### Community 4 - "RL Defender Training"
Cohesion: 0.17
Nodes (21): collect_environment_metadata(), configure_torch_runtime(), evaluate_model(), main(), make_env_fn(), _package_version(), parse_args(), train_rl_defender.py — Entrenamiento de agente defensor RL sobre CICIDS2017. (+13 more)

### Community 5 - "Canonical Schema Setup"
Cohesion: 0.12
Nodes (17): batched_predict(), load_model(), main(), maybe_convert_time_units(), CanonicalResult, get_canonical_feature_names(), get_observation_feature_names(), map_to_canonical() (+9 more)

### Community 6 - "Graphify Auto Update"
Cohesion: 0.25
Nodes (16): _has_structural_patch diff analyzer, _is_semantic_source path classifier, _rebuild_code_graph caller, decide_and_run(), _diff_range(), _has_structural_patch(), _is_code(), _is_ignored() (+8 more)

### Community 7 - "RL Environment Config"
Cohesion: 0.15
Nodes (7): Calcula la recompensa en función de la etiqueta real, la acción         y la co, Entorno RL para un defensor que decide PERMIT/BLOCK sobre muestras etiquetadas., RLDatasetDefenderEnv, test_reward_logic_tp_fp_tn_fn(), test_unknown_label_reward(), test_env_initialization(), test_env_step_and_reset()

### Community 8 - "Data Load Tools"
Cohesion: 0.15
Nodes (17): CanonicalResult dataclass, FEATURES_CANON (76 canonical network flow features), NSL_KDD_TO_CANON mapping dict (partial), map_to_canonical function, Missingness Mask Design (observation = features + mask), load_nsl_kdd_binary function, batched_predict in v1, load_model (QRDQN/DQN fallback) in v1 (+9 more)

### Community 9 - "Leave One CSV Out"
Cohesion: 0.25
Nodes (13): _build_aggregate_results(), _compute_reward_total(), evaluate_model_direct(), main(), make_env_fn(), _metrics_from_confusion(), parse_args(), validate_leave_one_csv_out.py — Validación leave-one-exact-CSV-out para CICIDS20 (+5 more)

### Community 10 - "Hyperparams & Rewards"
Cohesion: 0.22
Nodes (11): Reward Config Schema (tp/fp/fn/omission), RLDatasetDefenderEnv gymnasium environment, _compute_reward, Reward Config (tp/fp/fn/omission), test_reward_logic_tp_fp_tn_fn, test_unknown_label_reward, test_env_initialization, test_env_step_and_reset (+3 more)

### Community 11 - "Tensorboard Exports"
Cohesion: 0.46
Nodes (7): _find_event_dirs(), main(), parse_args(), _plot_scalar(), _read_scalars(), _safe_filename(), _update_artifact_manifest()

### Community 12 - "Hyperparam Tuning"
Cohesion: 0.36
Nodes (7): _evaluate_f1(), main(), objective(), parse_args(), tune_hparams.py -- Optimizacion de hiperparametros con Optuna para QRDQN sobre C, Evaluate model and return F1 score for attack class., Optuna objective: train QRDQN with suggested hparams, return F1 attack.

### Community 13 - "State of Art Research"
Cohesion: 0.4
Nodes (5): C:/Users/Rivero/Desktop/TFG_CYBER_AI/Research/Initial Research for Report - Chapter State Of The Art/RESEARCH_INDEX.md, C:/Users/Rivero/Desktop/TFG_CYBER_AI/Research/Initial Research for Report - Chapter State Of The Art/Research1.md, C:/Users/Rivero/Desktop/TFG_CYBER_AI/Research/Initial Research for Report - Chapter State Of The Art/Research2.md, C:/Users/Rivero/Desktop/TFG_CYBER_AI/Research/Initial Research for Report - Chapter State Of The Art/Research3.md, C:/Users/Rivero/Desktop/TFG_CYBER_AI/Research/Initial Research for Report - Chapter State Of The Art/report-source-map.md

### Community 14 - "Experimental Boards"
Cohesion: 0.5
Nodes (4): Gymnasium Environment: 152-Dim Observation (76 features + 76 mask), Experimental Design Board (Big Picture Pipeline), Branch A: Data Efficiency (100k-2M training budgets), Branch E: Strict Split Evaluation Ladder

### Community 15 - "QRDQN Methods"
Cohesion: 0.5
Nodes (4): C:/Users/Rivero/Desktop/TFG_CYBER_AI/Research/Initial Research for Report - Chapter State Of The Art/report-qrdqn-deep-distributional-rl.md, C:/Users/Rivero/Desktop/TFG_CYBER_AI/Research/Initial Research for Report - Chapter State Of The Art/report-reward-and-cost-sensitive-design-dossier.md, Cost-Sensitive Reward Function, QRDQN

### Community 16 - "Features Map"
Cohesion: 0.67
Nodes (3): CICIDS2017_TO_CANON mapping dict, FLOWMETER_PY_TO_CANON mapping in v1, FLOWMETER_PY_TO_CANON mapping in v2

### Community 17 - "NSLKDD Experiments"
Cohesion: 0.67
Nodes (3): NSL-KDD E01 DQN Experiment, NSL-KDD Historical Experiments, NSL-KDD E02 Random Forest Experiment

## Knowledge Gaps
- **75 isolated node(s):** `predict_real_traffic_v2.py — Robust Phase 2 offline inference pipeline.  Loads`, `If time columns look like seconds (median < 1), convert to microseconds.`, `Load QRDQN model, falling back to DQN if sb3_contrib is unavailable.`, `Run model.predict in batches to avoid OOM on large flow CSVs.`, `Compute z-score diagnostics on scaled features (first _N_CANON dims only).` (+70 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **4 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `map_to_canonical()` connect `Canonical Schema Setup` to `CICIDS2017 Preprocessing`, `Inference Diagnostics`?**
  _High betweenness centrality (0.233) - this node is a cross-community bridge._
- **Why does `main()` connect `Inference Diagnostics` to `Canonical Schema Setup`?**
  _High betweenness centrality (0.180) - this node is a cross-community bridge._
- **Why does `_prepare_cicids_features()` connect `CICIDS2017 Preprocessing` to `Canonical Schema Setup`?**
  _High betweenness centrality (0.140) - this node is a cross-community bridge._
- **Are the 6 inferred relationships involving `RLDatasetDefenderEnv` (e.g. with `ProgressCallback` and `_evaluate_f1()`) actually correct?**
  _`RLDatasetDefenderEnv` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `load_cicids2017_split()` (e.g. with `main()` and `main()`) actually correct?**
  _`load_cicids2017_split()` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `main()` (e.g. with `map_to_canonical()` and `apply_percentile_clipping()`) actually correct?**
  _`main()` has 3 INFERRED edges - model-reasoned connections that need verification._
- **What connects `predict_real_traffic_v2.py — Robust Phase 2 offline inference pipeline.  Loads`, `If time columns look like seconds (median < 1), convert to microseconds.`, `Load QRDQN model, falling back to DQN if sb3_contrib is unavailable.` to the rest of the system?**
  _75 weakly-connected nodes found - possible documentation gaps or missing edges._