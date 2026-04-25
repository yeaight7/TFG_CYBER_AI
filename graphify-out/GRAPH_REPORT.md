# Graph Report - TFG_CYBER_AI  (2026-04-25)

## Corpus Check
- 15 files · ~221,776 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 199 nodes · 281 edges · 55 communities detected
- Extraction: 78% EXTRACTED · 22% INFERRED · 0% AMBIGUOUS · INFERRED: 63 edges (avg confidence: 0.65)
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

## God Nodes (most connected - your core abstractions)
1. `RLDatasetDefenderEnv` - 26 edges
2. `CICIDSLoadConfig` - 25 edges
3. `main()` - 10 edges
4. `main()` - 10 edges
5. `decide_and_run()` - 10 edges
6. `_prepare_cicids_features()` - 9 edges
7. `load_cicids2017_binary()` - 9 edges
8. `load_cicids2017_split()` - 8 edges
9. `ProgressCallback` - 8 edges
10. `map_to_canonical()` - 7 edges

## Surprising Connections (you probably didn't know these)
- `Entrena un RandomForestClassifier sobre NSL-KDD y devuelve el modelo.` --uses--> `CICIDSLoadConfig`  [INFERRED]
  src\baseline_random_forest.py → src/load_cicids2017.py
- `Evalúa el Random Forest y muestra matriz de confusión + informe de clasificación` --uses--> `CICIDSLoadConfig`  [INFERRED]
  src\baseline_random_forest.py → src/load_cicids2017.py
- `main()` --calls--> `map_to_canonical()`  [INFERRED]
  scripts/predict_real_traffic_v2.py → src/canonical_schema.py
- `main()` --calls--> `map_to_canonical()`  [INFERRED]
  scripts/predict_real_traffic.py → src/canonical_schema.py
- `tune_hparams.py -- Optimizacion de hiperparametros con Optuna para QRDQN sobre C` --uses--> `CICIDSLoadConfig`  [INFERRED]
  src\tune_hparams.py → src/load_cicids2017.py

## Hyperedges (group relationships)
- **Canonical Observation Pipeline** — concept_canonical_flow_schema, concept_missingness_mask, concept_observation_vector_152, concept_cicids2017_dataset [EXTRACTED 1.00]
- **Phase 2 Lab Workflow** — concept_phase_2_offline_inference, concept_private_lab_workflow, concept_robust_v2_inference_pipeline, concept_domain_shift_diagnostics [EXTRACTED 1.00]
- **Historical NSL-KDD Track** — concept_historical_nsl_kdd_branch, concept_nsl_kdd_dataset, concept_random_forest_baseline, concept_reward_shaping [EXTRACTED 1.00]

## Communities

### Community 0 - "Community 0"
Cohesion: 0.14
Nodes (17): Calcula la recompensa en función de la etiqueta real, la acción         y la con, Entorno RL para un defensor que decide PERMIT/BLOCK sobre muestras etiquetadas., RLDatasetDefenderEnv, evaluate_model(), main(), make_env_fn(), parse_args(), train_rl_defender.py — Entrenamiento de agente defensor RL sobre CICIDS2017. (+9 more)

### Community 1 - "Community 1"
Cohesion: 0.15
Nodes (23): _clean_rows(), _coerce_numeric_features(), _drop_identifier_like_columns(), _find_label_column(), list_cicids2017_csv_files(), _list_csv_files(), _load_all_csvs(), _load_and_process_csv_paths() (+15 more)

### Community 2 - "Community 2"
Cohesion: 0.14
Nodes (18): batched_predict(), compute_diagnostics(), compute_truth_metrics(), load_model(), main(), maybe_convert_time_units(), parse_args(), predict_real_traffic_v2.py — Robust Phase 2 offline inference pipeline.  Loads (+10 more)

### Community 3 - "Community 3"
Cohesion: 0.19
Nodes (15): BaseCallback, CICIDSLoadConfig, load_cicids2017_split(), Unified CICIDS2017 loader with split-mode and preset support.      Parameters, check_a_direct_eval(), check_b_shuffled_labels(), check_c_csv_split(), main() (+7 more)

### Community 4 - "Community 4"
Cohesion: 0.24
Nodes (14): decide_and_run(), _diff_range(), _has_structural_patch(), _is_code(), _is_ignored(), _is_semantic_source(), main(), _normalize() (+6 more)

### Community 5 - "Community 5"
Cohesion: 0.17
Nodes (13): CanonicalResult, get_canonical_feature_names(), get_observation_feature_names(), map_to_canonical(), canonical_schema.py — Definición formal del esquema canónico de features (FEATUR, Resultado de mapear un DataFrame al esquema canónico., Mapea un DataFrame al esquema canónico de features.      Parameters     -----, Devuelve la lista de nombres de features canónicas (sin máscara). (+5 more)

### Community 6 - "Community 6"
Cohesion: 0.25
Nodes (13): _build_aggregate_results(), _compute_reward_total(), evaluate_model_direct(), main(), make_env_fn(), _metrics_from_confusion(), parse_args(), validate_leave_one_csv_out.py — Validación leave-one-exact-CSV-out para CICIDS20 (+5 more)

### Community 7 - "Community 7"
Cohesion: 0.47
Nodes (5): evaluate_random_forest(), main(), Entrena un RandomForestClassifier sobre NSL-KDD y devuelve el modelo., Evalúa el Random Forest y muestra matriz de confusión + informe de clasificación, train_random_forest()

### Community 8 - "Community 8"
Cohesion: 0.7
Nodes (4): batched_predict(), load_model(), main(), maybe_convert_time_units()

### Community 9 - "Community 9"
Cohesion: 0.7
Nodes (3): http_get(), main(), tcp_connect()

### Community 10 - "Community 10"
Cohesion: 1.0
Nodes (1): Calcula la recompensa en función de la etiqueta real, la acción         y la co

### Community 14 - "Community 14"
Cohesion: 1.0
Nodes (1): KDDTest+.txt

### Community 15 - "Community 15"
Cohesion: 1.0
Nodes (1): KDDTest-21.txt

### Community 16 - "Community 16"
Cohesion: 1.0
Nodes (1): KDDTrain+.txt

### Community 17 - "Community 17"
Cohesion: 1.0
Nodes (1): KDDTrain+_20Percent.txt

### Community 18 - "Community 18"
Cohesion: 1.0
Nodes (1): KDDTest+.txt

### Community 19 - "Community 19"
Cohesion: 1.0
Nodes (1): KDDTest-21.txt

### Community 20 - "Community 20"
Cohesion: 1.0
Nodes (1): KDDTrain+.txt

### Community 21 - "Community 21"
Cohesion: 1.0
Nodes (1): KDDTrain+_20Percent.txt

### Community 22 - "Community 22"
Cohesion: 1.0
Nodes (1): AGENT_CONTEXT.md

### Community 23 - "Community 23"
Cohesion: 1.0
Nodes (1): DEFENSA_TFG_PROGRESO.md

### Community 24 - "Community 24"
Cohesion: 1.0
Nodes (1): DEFENSA_TFG_SCRIPT.md

### Community 25 - "Community 25"
Cohesion: 1.0
Nodes (1): gcp_lab.md

### Community 26 - "Community 26"
Cohesion: 1.0
Nodes (1): phase2_plan.md

### Community 27 - "Community 27"
Cohesion: 1.0
Nodes (1): README.md

### Community 28 - "Community 28"
Cohesion: 1.0
Nodes (1): results.md

### Community 29 - "Community 29"
Cohesion: 1.0
Nodes (1): nslkdd_experiments.md

### Community 30 - "Community 30"
Cohesion: 1.0
Nodes (1): README.md

### Community 31 - "Community 31"
Cohesion: 1.0
Nodes (1): report.pdf

### Community 32 - "Community 32"
Cohesion: 1.0
Nodes (1): KDDTest1.jpg

### Community 33 - "Community 33"
Cohesion: 1.0
Nodes (1): KDDTrain1.jpg

### Community 34 - "Community 34"
Cohesion: 1.0
Nodes (1): KDDTest1.jpg

### Community 35 - "Community 35"
Cohesion: 1.0
Nodes (1): KDDTrain1.jpg

### Community 36 - "Community 36"
Cohesion: 1.0
Nodes (1): RL Cyber Defender Project

### Community 37 - "Community 37"
Cohesion: 1.0
Nodes (1): Phase 1 Offline Training

### Community 38 - "Community 38"
Cohesion: 1.0
Nodes (1): Phase 2 Offline Inference

### Community 39 - "Community 39"
Cohesion: 1.0
Nodes (1): Canonical Flow Schema

### Community 40 - "Community 40"
Cohesion: 1.0
Nodes (1): Missingness Mask

### Community 41 - "Community 41"
Cohesion: 1.0
Nodes (1): 152-D Observation Vector

### Community 42 - "Community 42"
Cohesion: 1.0
Nodes (1): Anti-Leakage Policy

### Community 43 - "Community 43"
Cohesion: 1.0
Nodes (1): CICIDS2017 Dataset

### Community 44 - "Community 44"
Cohesion: 1.0
Nodes (1): NSL-KDD Dataset

### Community 45 - "Community 45"
Cohesion: 1.0
Nodes (1): QRDQN Training Pipeline

### Community 46 - "Community 46"
Cohesion: 1.0
Nodes (1): RLDatasetDefenderEnv

### Community 47 - "Community 47"
Cohesion: 1.0
Nodes (1): Validation Suite

### Community 48 - "Community 48"
Cohesion: 1.0
Nodes (1): Leave-One-Exact-CSV-Out Validation

### Community 49 - "Community 49"
Cohesion: 1.0
Nodes (1): Robust v2 Inference Pipeline

### Community 50 - "Community 50"
Cohesion: 1.0
Nodes (1): Domain Shift Diagnostics

### Community 51 - "Community 51"
Cohesion: 1.0
Nodes (1): Reward Shaping

### Community 52 - "Community 52"
Cohesion: 1.0
Nodes (1): Private Lab Workflow

### Community 53 - "Community 53"
Cohesion: 1.0
Nodes (1): Artifact-Backed Results

### Community 54 - "Community 54"
Cohesion: 1.0
Nodes (1): Historical NSL-KDD Branch

### Community 55 - "Community 55"
Cohesion: 1.0
Nodes (1): Random Forest Baseline

### Community 56 - "Community 56"
Cohesion: 1.0
Nodes (1): Active Blocking Future Work

### Community 57 - "Community 57"
Cohesion: 1.0
Nodes (1): Documentation Map

## Knowledge Gaps
- **71 isolated node(s):** `scaling_utils.py — Clipping utilities for outlier handling in RL inference pipel`, `Clamp each feature to its training percentile range [p_low, p_high].      Appl`, `Clamp scaled features to [-max_z, +max_z].      Applied to scaled features **a`, `Carga NSL-KDD desde Kaggle (hassan06/nslkdd), lo preprocesa y devuelve:`, `canonical_schema.py — Definición formal del esquema canónico de features (FEATUR` (+66 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 10`** (1 nodes): `Calcula la recompensa en función de la etiqueta real, la acción         y la co`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 14`** (1 nodes): `KDDTest+.txt`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 15`** (1 nodes): `KDDTest-21.txt`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 16`** (1 nodes): `KDDTrain+.txt`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 17`** (1 nodes): `KDDTrain+_20Percent.txt`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 18`** (1 nodes): `KDDTest+.txt`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 19`** (1 nodes): `KDDTest-21.txt`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 20`** (1 nodes): `KDDTrain+.txt`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 21`** (1 nodes): `KDDTrain+_20Percent.txt`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 22`** (1 nodes): `AGENT_CONTEXT.md`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 23`** (1 nodes): `DEFENSA_TFG_PROGRESO.md`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 24`** (1 nodes): `DEFENSA_TFG_SCRIPT.md`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 25`** (1 nodes): `gcp_lab.md`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 26`** (1 nodes): `phase2_plan.md`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 27`** (1 nodes): `README.md`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 28`** (1 nodes): `results.md`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 29`** (1 nodes): `nslkdd_experiments.md`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 30`** (1 nodes): `README.md`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 31`** (1 nodes): `report.pdf`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 32`** (1 nodes): `KDDTest1.jpg`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 33`** (1 nodes): `KDDTrain1.jpg`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 34`** (1 nodes): `KDDTest1.jpg`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 35`** (1 nodes): `KDDTrain1.jpg`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 36`** (1 nodes): `RL Cyber Defender Project`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 37`** (1 nodes): `Phase 1 Offline Training`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 38`** (1 nodes): `Phase 2 Offline Inference`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 39`** (1 nodes): `Canonical Flow Schema`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 40`** (1 nodes): `Missingness Mask`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 41`** (1 nodes): `152-D Observation Vector`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 42`** (1 nodes): `Anti-Leakage Policy`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 43`** (1 nodes): `CICIDS2017 Dataset`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 44`** (1 nodes): `NSL-KDD Dataset`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 45`** (1 nodes): `QRDQN Training Pipeline`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 46`** (1 nodes): `RLDatasetDefenderEnv`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 47`** (1 nodes): `Validation Suite`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 48`** (1 nodes): `Leave-One-Exact-CSV-Out Validation`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 49`** (1 nodes): `Robust v2 Inference Pipeline`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 50`** (1 nodes): `Domain Shift Diagnostics`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 51`** (1 nodes): `Reward Shaping`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 52`** (1 nodes): `Private Lab Workflow`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 53`** (1 nodes): `Artifact-Backed Results`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 54`** (1 nodes): `Historical NSL-KDD Branch`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 55`** (1 nodes): `Random Forest Baseline`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 56`** (1 nodes): `Active Blocking Future Work`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 57`** (1 nodes): `Documentation Map`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `map_to_canonical()` connect `Community 5` to `Community 8`, `Community 1`, `Community 2`?**
  _High betweenness centrality (0.206) - this node is a cross-community bridge._
- **Why does `CICIDSLoadConfig` connect `Community 3` to `Community 0`, `Community 1`, `Community 6`, `Community 7`?**
  _High betweenness centrality (0.178) - this node is a cross-community bridge._
- **Why does `main()` connect `Community 2` to `Community 5`?**
  _High betweenness centrality (0.169) - this node is a cross-community bridge._
- **Are the 17 inferred relationships involving `RLDatasetDefenderEnv` (e.g. with `tune_hparams.py -- Optimizacion de hiperparametros con Optuna para QRDQN sobre C` and `Evaluate model and return F1 score for attack class.`) actually correct?**
  _`RLDatasetDefenderEnv` has 17 INFERRED edges - model-reasoned connections that need verification._
- **Are the 20 inferred relationships involving `CICIDSLoadConfig` (e.g. with `tune_hparams.py -- Optimizacion de hiperparametros con Optuna para QRDQN sobre C` and `Evaluate model and return F1 score for attack class.`) actually correct?**
  _`CICIDSLoadConfig` has 20 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `main()` (e.g. with `list_cicids2017_csv_files()` and `CICIDSLoadConfig`) actually correct?**
  _`main()` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `main()` (e.g. with `map_to_canonical()` and `apply_percentile_clipping()`) actually correct?**
  _`main()` has 3 INFERRED edges - model-reasoned connections that need verification._