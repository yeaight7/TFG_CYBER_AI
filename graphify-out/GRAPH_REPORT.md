# Graph Report - C:\Users\Rivero\Desktop\TFG_CYBER_AI  (2026-05-03)

## Corpus Check
- 19 files · ~21,761 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 201 nodes · 252 edges · 59 communities detected
- Extraction: 87% EXTRACTED · 13% INFERRED · 0% AMBIGUOUS · INFERRED: 32 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 35|Community 35]]
- [[_COMMUNITY_Community 36|Community 36]]
- [[_COMMUNITY_Community 37|Community 37]]
- [[_COMMUNITY_Community 38|Community 38]]
- [[_COMMUNITY_Community 39|Community 39]]
- [[_COMMUNITY_Community 40|Community 40]]
- [[_COMMUNITY_Community 41|Community 41]]
- [[_COMMUNITY_Community 42|Community 42]]
- [[_COMMUNITY_Community 43|Community 43]]
- [[_COMMUNITY_Community 44|Community 44]]
- [[_COMMUNITY_Community 45|Community 45]]
- [[_COMMUNITY_Community 46|Community 46]]
- [[_COMMUNITY_Community 47|Community 47]]
- [[_COMMUNITY_Community 48|Community 48]]
- [[_COMMUNITY_Community 49|Community 49]]
- [[_COMMUNITY_Community 50|Community 50]]
- [[_COMMUNITY_Community 51|Community 51]]
- [[_COMMUNITY_Community 52|Community 52]]
- [[_COMMUNITY_Community 53|Community 53]]
- [[_COMMUNITY_Community 54|Community 54]]
- [[_COMMUNITY_Community 55|Community 55]]
- [[_COMMUNITY_Community 56|Community 56]]
- [[_COMMUNITY_Community 57|Community 57]]
- [[_COMMUNITY_Community 58|Community 58]]

## God Nodes (most connected - your core abstractions)
1. `RLDatasetDefenderEnv` - 24 edges
2. `CICIDSLoadConfig` - 22 edges
3. `decide_and_run()` - 10 edges
4. `_prepare_cicids_features()` - 8 edges
5. `ProgressCallback` - 8 edges
6. `main()` - 7 edges
7. `_load_and_process_csv_paths()` - 7 edges
8. `load_cicids2017_binary()` - 7 edges
9. `load_cicids2017_csv_split()` - 6 edges
10. `load_cicids2017_exact_csv_split()` - 6 edges

## Surprising Connections (you probably didn't know these)
- `Entrena un RandomForestClassifier sobre NSL-KDD y devuelve el modelo.` --uses--> `CICIDSLoadConfig`  [INFERRED]
  src/baseline_random_forest.py → C:\Users\Rivero\Desktop\TFG_CYBER_AI\src\load_cicids2017.py
- `Evalúa el Random Forest y muestra matriz de confusión + informe de clasificación` --uses--> `CICIDSLoadConfig`  [INFERRED]
  src/baseline_random_forest.py → C:\Users\Rivero\Desktop\TFG_CYBER_AI\src\load_cicids2017.py
- `train_rl_defender.py — Entrenamiento de agente defensor RL sobre CICIDS2017.` --uses--> `CICIDSLoadConfig`  [INFERRED]
  src/train_rl_defender.py → C:\Users\Rivero\Desktop\TFG_CYBER_AI\src\load_cicids2017.py
- `Devuelve una función creadora de entornos para DummyVecEnv.` --uses--> `CICIDSLoadConfig`  [INFERRED]
  src/train_rl_defender.py → C:\Users\Rivero\Desktop\TFG_CYBER_AI\src\load_cicids2017.py
- `Evalúa el agente sobre test set.     Devuelve dict con métricas clave.` --uses--> `CICIDSLoadConfig`  [INFERRED]
  src/train_rl_defender.py → C:\Users\Rivero\Desktop\TFG_CYBER_AI\src\load_cicids2017.py

## Hyperedges (group relationships)
- **Canonical Observation Pipeline** — concept_canonical_flow_schema, concept_76_canonical_features, concept_missingness_mask, concept_observation_vector_152, src_canonical_schema_py [EXTRACTED 1.00]
- **CICIDS2017 Training and Validation Pipeline** — concept_cicids2017_dataset, concept_qrdqn_training_pipeline, concept_validation_suite, src_load_cicids2017_py, src_train_rl_defender_py, src_validate_checks_py [EXTRACTED 1.00]
- **Phase 2 Lab Inference Workflow** — concept_phase2_offline_inference, concept_private_lab_workflow, concept_robust_v2_inference_pipeline, concept_domain_shift_risk, scripts_predict_real_traffic_v2_py [EXTRACTED 1.00]
- **Documentation and Defense Alignment** — doc_docs_readme, doc_results, doc_defensa_progreso, doc_defensa_script, concept_artifact_backed_results, concept_defense_positioning [EXTRACTED 1.00]

## Communities

### Community 0 - "Community 0"
Cohesion: 0.14
Nodes (25): _clean_rows(), _coerce_numeric_features(), _drop_identifier_like_columns(), _find_label_column(), list_cicids2017_csv_files(), _list_csv_files(), _load_all_csvs(), _load_and_process_csv_paths() (+17 more)

### Community 1 - "Community 1"
Cohesion: 0.15
Nodes (10): Calcula la recompensa en función de la etiqueta real, la acción         y la co, Entorno RL para un defensor que decide PERMIT/BLOCK sobre muestras etiquetadas., RLDatasetDefenderEnv, evaluate_model(), main(), make_env_fn(), parse_args(), train_rl_defender.py — Entrenamiento de agente defensor RL sobre CICIDS2017. (+2 more)

### Community 2 - "Community 2"
Cohesion: 0.21
Nodes (13): BaseCallback, CICIDSLoadConfig, check_a_direct_eval(), check_b_shuffled_labels(), check_c_csv_split(), main(), parse_args(), ProgressCallback (+5 more)

### Community 3 - "Community 3"
Cohesion: 0.34
Nodes (13): decide_and_run(), _diff_range(), _has_structural_patch(), _is_code(), _is_ignored(), _is_semantic_source(), main(), _normalize() (+5 more)

### Community 4 - "Community 4"
Cohesion: 0.21
Nodes (13): batched_predict(), compute_diagnostics(), compute_truth_metrics(), load_model(), main(), maybe_convert_time_units(), parse_args(), predict_real_traffic_v2.py — Robust Phase 2 offline inference pipeline.  Loads (+5 more)

### Community 5 - "Community 5"
Cohesion: 0.25
Nodes (13): _build_aggregate_results(), _compute_reward_total(), evaluate_model_direct(), main(), make_env_fn(), _metrics_from_confusion(), parse_args(), validate_leave_one_csv_out.py — Validación leave-one-exact-CSV-out para CICIDS20 (+5 more)

### Community 6 - "Community 6"
Cohesion: 0.22
Nodes (9): CanonicalResult, get_canonical_feature_names(), get_observation_feature_names(), map_to_canonical(), canonical_schema.py — Definición formal del esquema canónico de features (FEATUR, Resultado de mapear un DataFrame al esquema canónico., Mapea un DataFrame al esquema canónico de features.      Parameters     -----, Devuelve la lista de nombres de features canónicas (sin máscara). (+1 more)

### Community 7 - "Community 7"
Cohesion: 0.36
Nodes (7): _evaluate_f1(), main(), objective(), parse_args(), tune_hparams.py -- Optimizacion de hiperparametros con Optuna para QRDQN sobre C, Evaluate model and return F1 score for attack class., Optuna objective: train QRDQN with suggested hparams, return F1 attack.

### Community 8 - "Community 8"
Cohesion: 0.47
Nodes (5): evaluate_random_forest(), main(), Entrena un RandomForestClassifier sobre NSL-KDD y devuelve el modelo., Evalúa el Random Forest y muestra matriz de confusión + informe de clasificación, train_random_forest()

### Community 9 - "Community 9"
Cohesion: 0.33
Nodes (5): apply_percentile_clipping(), apply_z_clipping(), scaling_utils.py — Clipping utilities for outlier handling in RL inference pipel, Clamp each feature to its training percentile range [p_low, p_high].      Appl, Clamp scaled features to [-max_z, +max_z].      Applied to scaled features **a

### Community 10 - "Community 10"
Cohesion: 0.7
Nodes (4): batched_predict(), load_model(), main(), maybe_convert_time_units()

### Community 11 - "Community 11"
Cohesion: 0.67
Nodes (3): _ensure_dataset_local_dir(), load_nsl_kdd_binary(), Carga NSL-KDD desde Kaggle (hassan06/nslkdd), lo preprocesa y devuelve:

### Community 12 - "Community 12"
Cohesion: 0.5
Nodes (1): GraphifyAutoUpdateSemanticSourceTests

### Community 13 - "Community 13"
Cohesion: 0.67
Nodes (0): 

### Community 14 - "Community 14"
Cohesion: 0.67
Nodes (0): 

### Community 15 - "Community 15"
Cohesion: 0.67
Nodes (0): 

### Community 16 - "Community 16"
Cohesion: 0.67
Nodes (0): 

### Community 17 - "Community 17"
Cohesion: 1.0
Nodes (0): 

### Community 18 - "Community 18"
Cohesion: 1.0
Nodes (1): Carga NSL-KDD desde Kaggle (hassan06/nslkdd), lo preprocesa y devuelve:

### Community 19 - "Community 19"
Cohesion: 1.0
Nodes (1): AGENTS.md Project Operating Rules

### Community 20 - "Community 20"
Cohesion: 1.0
Nodes (1): Public README Project Overview

### Community 21 - "Community 21"
Cohesion: 1.0
Nodes (1): Project-Wide Technical Source of Truth

### Community 22 - "Community 22"
Cohesion: 1.0
Nodes (1): Documentation Map

### Community 23 - "Community 23"
Cohesion: 1.0
Nodes (1): Phase 2 Context and Guardrails

### Community 24 - "Community 24"
Cohesion: 1.0
Nodes (1): Artifact-Backed Results Snapshot

### Community 25 - "Community 25"
Cohesion: 1.0
Nodes (1): Phase 2 Execution Plan

### Community 26 - "Community 26"
Cohesion: 1.0
Nodes (1): Private Lab Setup Guide

### Community 27 - "Community 27"
Cohesion: 1.0
Nodes (1): Spanish Defense Progress Notes

### Community 28 - "Community 28"
Cohesion: 1.0
Nodes (1): Spanish Oral Defense Script

### Community 29 - "Community 29"
Cohesion: 1.0
Nodes (1): Python Runtime Dependencies

### Community 30 - "Community 30"
Cohesion: 1.0
Nodes (1): RL Cyber Defender Project

### Community 31 - "Community 31"
Cohesion: 1.0
Nodes (1): Phase 1 Offline Training and Validation

### Community 32 - "Community 32"
Cohesion: 1.0
Nodes (1): Phase 2 Offline Inference

### Community 33 - "Community 33"
Cohesion: 1.0
Nodes (1): Binary PERMIT/BLOCK Defender Actions

### Community 34 - "Community 34"
Cohesion: 1.0
Nodes (1): Canonical Flow Schema

### Community 35 - "Community 35"
Cohesion: 1.0
Nodes (1): 76 Canonical Flow Features

### Community 36 - "Community 36"
Cohesion: 1.0
Nodes (1): Missingness Mask

### Community 37 - "Community 37"
Cohesion: 1.0
Nodes (1): 152-D Observation Vector

### Community 38 - "Community 38"
Cohesion: 1.0
Nodes (1): Dataset Adapter Contract

### Community 39 - "Community 39"
Cohesion: 1.0
Nodes (1): Anti-Leakage Policy

### Community 40 - "Community 40"
Cohesion: 1.0
Nodes (1): CICIDS2017 Primary Dataset

### Community 41 - "Community 41"
Cohesion: 1.0
Nodes (1): NSL-KDD Historical Benchmark

### Community 42 - "Community 42"
Cohesion: 1.0
Nodes (1): QRDQN Training Pipeline

### Community 43 - "Community 43"
Cohesion: 1.0
Nodes (1): Current Reward Configuration

### Community 44 - "Community 44"
Cohesion: 1.0
Nodes (1): Validation Suite

### Community 45 - "Community 45"
Cohesion: 1.0
Nodes (1): Check A Direct Evaluation

### Community 46 - "Community 46"
Cohesion: 1.0
Nodes (1): Check B Shuffled-Label Anti-Leakage

### Community 47 - "Community 47"
Cohesion: 1.0
Nodes (1): Check C Hard CSV/Day Split

### Community 48 - "Community 48"
Cohesion: 1.0
Nodes (1): Leave-One-Exact-CSV-Out Validation

### Community 49 - "Community 49"
Cohesion: 1.0
Nodes (1): Artifact-Backed Results

### Community 50 - "Community 50"
Cohesion: 1.0
Nodes (1): C03 Best Historical CICIDS2017 Run

### Community 51 - "Community 51"
Cohesion: 1.0
Nodes (1): Robust v2 Inference Pipeline

### Community 52 - "Community 52"
Cohesion: 1.0
Nodes (1): Domain Shift Risk

### Community 53 - "Community 53"
Cohesion: 1.0
Nodes (1): Private Lab Workflow

### Community 54 - "Community 54"
Cohesion: 1.0
Nodes (1): Two-VM Private Lab Topology

### Community 55 - "Community 55"
Cohesion: 1.0
Nodes (1): Reproducible Run Artifacts

### Community 56 - "Community 56"
Cohesion: 1.0
Nodes (1): Active Blocking Future Work

### Community 57 - "Community 57"
Cohesion: 1.0
Nodes (1): Honest Defense Positioning

### Community 58 - "Community 58"
Cohesion: 1.0
Nodes (1): Graphify Corpus Policy

## Knowledge Gaps
- **67 isolated node(s):** `predict_real_traffic_v2.py — Robust Phase 2 offline inference pipeline.  Loads`, `If time columns look like seconds (median < 1), convert to microseconds.`, `Load QRDQN model, falling back to DQN if sb3_contrib is unavailable.`, `Run model.predict in batches to avoid OOM on large flow CSVs.`, `Compute z-score diagnostics on scaled features (first _N_CANON dims only).` (+62 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 17`** (2 nodes): `test_load_cicids2017.py`, `test_prepare_cicids_features_binary_labels()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 18`** (1 nodes): `Carga NSL-KDD desde Kaggle (hassan06/nslkdd), lo preprocesa y devuelve:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 19`** (1 nodes): `AGENTS.md Project Operating Rules`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 20`** (1 nodes): `Public README Project Overview`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 21`** (1 nodes): `Project-Wide Technical Source of Truth`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 22`** (1 nodes): `Documentation Map`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 23`** (1 nodes): `Phase 2 Context and Guardrails`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 24`** (1 nodes): `Artifact-Backed Results Snapshot`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 25`** (1 nodes): `Phase 2 Execution Plan`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 26`** (1 nodes): `Private Lab Setup Guide`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 27`** (1 nodes): `Spanish Defense Progress Notes`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 28`** (1 nodes): `Spanish Oral Defense Script`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 29`** (1 nodes): `Python Runtime Dependencies`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 30`** (1 nodes): `RL Cyber Defender Project`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 31`** (1 nodes): `Phase 1 Offline Training and Validation`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 32`** (1 nodes): `Phase 2 Offline Inference`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 33`** (1 nodes): `Binary PERMIT/BLOCK Defender Actions`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 34`** (1 nodes): `Canonical Flow Schema`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 35`** (1 nodes): `76 Canonical Flow Features`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 36`** (1 nodes): `Missingness Mask`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 37`** (1 nodes): `152-D Observation Vector`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 38`** (1 nodes): `Dataset Adapter Contract`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 39`** (1 nodes): `Anti-Leakage Policy`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 40`** (1 nodes): `CICIDS2017 Primary Dataset`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 41`** (1 nodes): `NSL-KDD Historical Benchmark`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 42`** (1 nodes): `QRDQN Training Pipeline`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 43`** (1 nodes): `Current Reward Configuration`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 44`** (1 nodes): `Validation Suite`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 45`** (1 nodes): `Check A Direct Evaluation`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 46`** (1 nodes): `Check B Shuffled-Label Anti-Leakage`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 47`** (1 nodes): `Check C Hard CSV/Day Split`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 48`** (1 nodes): `Leave-One-Exact-CSV-Out Validation`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 49`** (1 nodes): `Artifact-Backed Results`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 50`** (1 nodes): `C03 Best Historical CICIDS2017 Run`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 51`** (1 nodes): `Robust v2 Inference Pipeline`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 52`** (1 nodes): `Domain Shift Risk`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 53`** (1 nodes): `Private Lab Workflow`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 54`** (1 nodes): `Two-VM Private Lab Topology`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 55`** (1 nodes): `Reproducible Run Artifacts`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 56`** (1 nodes): `Active Blocking Future Work`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 57`** (1 nodes): `Honest Defense Positioning`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 58`** (1 nodes): `Graphify Corpus Policy`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `CICIDSLoadConfig` connect `Community 2` to `Community 0`, `Community 1`, `Community 5`, `Community 7`, `Community 8`?**
  _High betweenness centrality (0.118) - this node is a cross-community bridge._
- **Why does `RLDatasetDefenderEnv` connect `Community 1` to `Community 2`, `Community 5`, `Community 7`?**
  _High betweenness centrality (0.061) - this node is a cross-community bridge._
- **Why does `validate_leave_one_csv_out.py — Validación leave-one-exact-CSV-out para CICIDS20` connect `Community 5` to `Community 1`, `Community 2`?**
  _High betweenness centrality (0.031) - this node is a cross-community bridge._
- **Are the 15 inferred relationships involving `RLDatasetDefenderEnv` (e.g. with `train_rl_defender.py — Entrenamiento de agente defensor RL sobre CICIDS2017.` and `Devuelve una función creadora de entornos para DummyVecEnv.`) actually correct?**
  _`RLDatasetDefenderEnv` has 15 INFERRED edges - model-reasoned connections that need verification._
- **Are the 17 inferred relationships involving `CICIDSLoadConfig` (e.g. with `Entrena un RandomForestClassifier sobre NSL-KDD y devuelve el modelo.` and `Evalúa el Random Forest y muestra matriz de confusión + informe de clasificación`) actually correct?**
  _`CICIDSLoadConfig` has 17 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `ProgressCallback` (e.g. with `RLDatasetDefenderEnv` and `CICIDSLoadConfig`) actually correct?**
  _`ProgressCallback` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `predict_real_traffic_v2.py — Robust Phase 2 offline inference pipeline.  Loads`, `If time columns look like seconds (median < 1), convert to microseconds.`, `Load QRDQN model, falling back to DQN if sb3_contrib is unavailable.` to the rest of the system?**
  _67 weakly-connected nodes found - possible documentation gaps or missing edges._