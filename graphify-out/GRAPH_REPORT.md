# Graph Report - C:\Users\Rivero\Desktop\TFG_CYBER_AI  (2026-04-12)

## Corpus Check

- 14 files · ~222,813 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary

- 194 nodes · 215 edges · 61 communities detected
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
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
- [[_COMMUNITY_Community 59|Community 59]]
- [[_COMMUNITY_Community 60|Community 60]]

## God Nodes (most connected - your core abstractions)

1. `decide_and_run()` - 10 edges
2. `RLDatasetDefenderEnv` - 9 edges
3. `_prepare_cicids_features()` - 8 edges
4. `main()` - 7 edges
5. `_load_and_process_csv_paths()` - 7 edges
6. `load_cicids2017_binary()` - 7 edges
7. `load_cicids2017_csv_split()` - 6 edges
8. `load_cicids2017_exact_csv_split()` - 6 edges
9. `ProgressCallback` - 6 edges
10. `main()` - 6 edges

## Surprising Connections (you probably didn't know these)

- None detected - all connections are within the same source files.

## Hyperedges (group relationships)

- **Canonical Observation Pipeline** — concept_canonical_flow_schema, concept_missingness_mask, concept_observation_vector_152, concept_cicids2017_dataset [EXTRACTED 1.00]
- **Phase 2 Lab Workflow** — concept_phase_2_offline_inference, concept_private_lab_workflow, concept_robust_v2_inference_pipeline, concept_domain_shift_diagnostics [EXTRACTED 1.00]
- **Historical NSL-KDD Track** — concept_historical_nsl_kdd_branch, concept_nsl_kdd_dataset, concept_random_forest_baseline, concept_reward_shaping [EXTRACTED 1.00]

## Communities

### Community 0 - "Community 0"

Cohesion: 0.14
Nodes (26): CICIDSLoadConfig, _clean_rows(), _coerce_numeric_features(), _drop_identifier_like_columns(), _find_label_column(), list_cicids2017_csv_files(), _list_csv_files(), _load_all_csvs() (+18 more)

### Community 1 - "Community 1"

Cohesion: 0.18
Nodes (12): BaseCallback, check_a_direct_eval(), check_b_shuffled_labels(), check_c_csv_split(), main(), parse_args(), ProgressCallback, validate_checks.py — Validación de resultados experimentales del agente RL.  I (+4 more)

### Community 2 - "Community 2"

Cohesion: 0.34
Nodes (13): decide_and_run(), _diff_range(), _has_structural_patch(), _is_code(), _is_ignored(), _is_semantic_source(), main(), _normalize() (+5 more)

### Community 3 - "Community 3"

Cohesion: 0.21
Nodes (13): batched_predict(), compute_diagnostics(), compute_truth_metrics(), load_model(), main(), maybe_convert_time_units(), parse_args(), predict_real_traffic_v2.py — Robust Phase 2 offline inference pipeline.  Loads (+5 more)

### Community 4 - "Community 4"

Cohesion: 0.25
Nodes (13): _build_aggregate_results(), _compute_reward_total(), evaluate_model_direct(), main(), make_env_fn(), _metrics_from_confusion(), parse_args(), validate_leave_one_csv_out.py — Validación leave-one-exact-CSV-out para CICIDS20 (+5 more)

### Community 5 - "Community 5"

Cohesion: 0.24
Nodes (3): Calcula la recompensa en función de la etiqueta real, la acción         y la co, Entorno RL para un defensor que decide PERMIT/BLOCK sobre muestras etiquetadas., RLDatasetDefenderEnv

### Community 6 - "Community 6"

Cohesion: 0.22
Nodes (9): CanonicalResult, get_canonical_feature_names(), get_observation_feature_names(), map_to_canonical(), canonical_schema.py — Definición formal del esquema canónico de features (FEATUR, Resultado de mapear un DataFrame al esquema canónico., Mapea un DataFrame al esquema canónico de features.      Parameters     -----, Devuelve la lista de nombres de features canónicas (sin máscara). (+1 more)

### Community 7 - "Community 7"

Cohesion: 0.36
Nodes (7): evaluate_model(), main(), make_env_fn(), parse_args(), train_rl_defender.py — Entrenamiento de agente defensor RL sobre CICIDS2017., Devuelve una función creadora de entornos para DummyVecEnv., Evalúa el agente sobre test set.     Devuelve dict con métricas clave.

### Community 8 - "Community 8"

Cohesion: 0.36
Nodes (7): _evaluate_f1(), main(), objective(), parse_args(), tune_hparams.py -- Optimizacion de hiperparametros con Optuna para QRDQN sobre C, Evaluate model and return F1 score for attack class., Optuna objective: train QRDQN with suggested hparams, return F1 attack.

### Community 9 - "Community 9"

Cohesion: 0.47
Nodes (5): evaluate_random_forest(), main(), Entrena un RandomForestClassifier sobre NSL-KDD y devuelve el modelo., Evalúa el Random Forest y muestra matriz de confusión + informe de clasificación, train_random_forest()

### Community 10 - "Community 10"

Cohesion: 0.33
Nodes (5): apply_percentile_clipping(), apply_z_clipping(), scaling_utils.py — Clipping utilities for outlier handling in RL inference pipel, Clamp each feature to its training percentile range [p_low, p_high].      Appl, Clamp scaled features to [-max_z, +max_z].      Applied to scaled features **a

### Community 11 - "Community 11"

Cohesion: 0.7
Nodes (4): batched_predict(), load_model(), main(), maybe_convert_time_units()

### Community 12 - "Community 12"

Cohesion: 0.6
Nodes (4): _download_nsl_kdd_via_kagglehub(), _ensure_dataset_local_dir(), load_nsl_kdd_binary(), Carga NSL-KDD desde Kaggle (hassan06/nslkdd), lo preprocesa y devuelve:

### Community 13 - "Community 13"

Cohesion: 0.83
Nodes (3): http_get(), main(), tcp_connect()

### Community 14 - "Community 14"

Cohesion: 1.0
Nodes (0):

### Community 15 - "Community 15"

Cohesion: 1.0
Nodes (0):

### Community 16 - "Community 16"

Cohesion: 1.0
Nodes (0):

### Community 17 - "Community 17"

Cohesion: 1.0
Nodes (0):

### Community 18 - "Community 18"

Cohesion: 1.0
Nodes (0):

### Community 19 - "Community 19"

Cohesion: 1.0
Nodes (0):

### Community 20 - "Community 20"

Cohesion: 1.0
Nodes (0):

### Community 21 - "Community 21"

Cohesion: 1.0
Nodes (0):

### Community 22 - "Community 22"

Cohesion: 1.0
Nodes (0):

### Community 23 - "Community 23"

Cohesion: 1.0
Nodes (0):

### Community 24 - "Community 24"

Cohesion: 1.0
Nodes (0):

### Community 25 - "Community 25"

Cohesion: 1.0
Nodes (0):

### Community 26 - "Community 26"

Cohesion: 1.0
Nodes (0):

### Community 27 - "Community 27"

Cohesion: 1.0
Nodes (0):

### Community 28 - "Community 28"

Cohesion: 1.0
Nodes (0):

### Community 29 - "Community 29"

Cohesion: 1.0
Nodes (0):

### Community 30 - "Community 30"

Cohesion: 1.0
Nodes (0):

### Community 31 - "Community 31"

Cohesion: 1.0
Nodes (0):

### Community 32 - "Community 32"

Cohesion: 1.0
Nodes (0):

### Community 33 - "Community 33"

Cohesion: 1.0
Nodes (0):

### Community 34 - "Community 34"

Cohesion: 1.0
Nodes (0):

### Community 35 - "Community 35"

Cohesion: 1.0
Nodes (0):

### Community 36 - "Community 36"

Cohesion: 1.0
Nodes (0):

### Community 37 - "Community 37"

Cohesion: 1.0
Nodes (0):

### Community 38 - "Community 38"

Cohesion: 1.0
Nodes (0):

### Community 39 - "Community 39"

Cohesion: 1.0
Nodes (1): RL Cyber Defender Project

### Community 40 - "Community 40"

Cohesion: 1.0
Nodes (1): Phase 1 Offline Training

### Community 41 - "Community 41"

Cohesion: 1.0
Nodes (1): Phase 2 Offline Inference

### Community 42 - "Community 42"

Cohesion: 1.0
Nodes (1): Canonical Flow Schema

### Community 43 - "Community 43"

Cohesion: 1.0
Nodes (1): Missingness Mask

### Community 44 - "Community 44"

Cohesion: 1.0
Nodes (1): 152-D Observation Vector

### Community 45 - "Community 45"

Cohesion: 1.0
Nodes (1): Anti-Leakage Policy

### Community 46 - "Community 46"

Cohesion: 1.0
Nodes (1): CICIDS2017 Dataset

### Community 47 - "Community 47"

Cohesion: 1.0
Nodes (1): NSL-KDD Dataset

### Community 48 - "Community 48"

Cohesion: 1.0
Nodes (1): QRDQN Training Pipeline

### Community 49 - "Community 49"

Cohesion: 1.0
Nodes (1): RLDatasetDefenderEnv

### Community 50 - "Community 50"

Cohesion: 1.0
Nodes (1): Validation Suite

### Community 51 - "Community 51"

Cohesion: 1.0
Nodes (1): Leave-One-Exact-CSV-Out Validation

### Community 52 - "Community 52"

Cohesion: 1.0
Nodes (1): Robust v2 Inference Pipeline

### Community 53 - "Community 53"

Cohesion: 1.0
Nodes (1): Domain Shift Diagnostics

### Community 54 - "Community 54"

Cohesion: 1.0
Nodes (1): Reward Shaping

### Community 55 - "Community 55"

Cohesion: 1.0
Nodes (1): Private Lab Workflow

### Community 56 - "Community 56"

Cohesion: 1.0
Nodes (1): Artifact-Backed Results

### Community 57 - "Community 57"

Cohesion: 1.0
Nodes (1): Historical NSL-KDD Branch

### Community 58 - "Community 58"

Cohesion: 1.0
Nodes (1): Random Forest Baseline

### Community 59 - "Community 59"

Cohesion: 1.0
Nodes (1): Active Blocking Future Work

### Community 60 - "Community 60"

Cohesion: 1.0
Nodes (1): Documentation Map

## Knowledge Gaps

- **64 isolated node(s):** `predict_real_traffic_v2.py — Robust Phase 2 offline inference pipeline.  Loads`, `If time columns look like seconds (median < 1), convert to microseconds.`, `Load QRDQN model, falling back to DQN if sb3_contrib is unavailable.`, `Run model.predict in batches to avoid OOM on large flow CSVs.`, `Compute z-score diagnostics on scaled features (first _N_CANON dims only).` (+59 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 14`** (1 nodes): `AGENTS.md`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 15`** (1 nodes): `README.md`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 16`** (1 nodes): `requirements.txt`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 17`** (1 nodes): `KDDTest+.txt`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 18`** (1 nodes): `KDDTest-21.txt`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 19`** (1 nodes): `KDDTrain+.txt`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 20`** (1 nodes): `KDDTrain+_20Percent.txt`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 21`** (1 nodes): `KDDTest+.txt`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 22`** (1 nodes): `KDDTest-21.txt`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 23`** (1 nodes): `KDDTrain+.txt`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 24`** (1 nodes): `KDDTrain+_20Percent.txt`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 25`** (1 nodes): `AGENT_CONTEXT.md`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 26`** (1 nodes): `DEFENSA_TFG_PROGRESO.md`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 27`** (1 nodes): `DEFENSA_TFG_SCRIPT.md`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 28`** (1 nodes): `gcp_lab.md`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 29`** (1 nodes): `phase2_plan.md`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 30`** (1 nodes): `README.md`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 31`** (1 nodes): `results.md`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 32`** (1 nodes): `nslkdd_experiments.md`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 33`** (1 nodes): `README.md`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 34`** (1 nodes): `report.pdf`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 35`** (1 nodes): `KDDTest1.jpg`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 36`** (1 nodes): `KDDTrain1.jpg`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 37`** (1 nodes): `KDDTest1.jpg`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 38`** (1 nodes): `KDDTrain1.jpg`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 39`** (1 nodes): `RL Cyber Defender Project`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 40`** (1 nodes): `Phase 1 Offline Training`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 41`** (1 nodes): `Phase 2 Offline Inference`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 42`** (1 nodes): `Canonical Flow Schema`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 43`** (1 nodes): `Missingness Mask`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 44`** (1 nodes): `152-D Observation Vector`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 45`** (1 nodes): `Anti-Leakage Policy`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 46`** (1 nodes): `CICIDS2017 Dataset`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 47`** (1 nodes): `NSL-KDD Dataset`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 48`** (1 nodes): `QRDQN Training Pipeline`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 49`** (1 nodes): `RLDatasetDefenderEnv`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 50`** (1 nodes): `Validation Suite`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 51`** (1 nodes): `Leave-One-Exact-CSV-Out Validation`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 52`** (1 nodes): `Robust v2 Inference Pipeline`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 53`** (1 nodes): `Domain Shift Diagnostics`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 54`** (1 nodes): `Reward Shaping`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 55`** (1 nodes): `Private Lab Workflow`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 56`** (1 nodes): `Artifact-Backed Results`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 57`** (1 nodes): `Historical NSL-KDD Branch`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 58`** (1 nodes): `Random Forest Baseline`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 59`** (1 nodes): `Active Blocking Future Work`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 60`** (1 nodes): `Documentation Map`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions

_Questions this graph is uniquely positioned to answer:_

- **What connects `predict_real_traffic_v2.py — Robust Phase 2 offline inference pipeline.  Loads`, `If time columns look like seconds (median < 1), convert to microseconds.`, `Load QRDQN model, falling back to DQN if sb3_contrib is unavailable.` to the rest of the system?**
  _64 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.14 - nodes in this community are weakly interconnected._
