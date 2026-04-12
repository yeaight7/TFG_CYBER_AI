# Graph Report - .  (2026-04-12)

## Corpus Check
- 38 files · ~393,223 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 180 nodes · 354 edges · 11 communities detected
- Extraction: 82% EXTRACTED · 18% INFERRED · 0% AMBIGUOUS · INFERRED: 63 edges (avg confidence: 0.67)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Environment and Tuning|Environment and Tuning]]
- [[_COMMUNITY_Project Docs and Phase Plan|Project Docs and Phase Plan]]
- [[_COMMUNITY_CICIDS2017 Loader|CICIDS2017 Loader]]
- [[_COMMUNITY_NSL-KDD History and Assets|NSL-KDD History and Assets]]
- [[_COMMUNITY_Phase 2 Inference Scripts|Phase 2 Inference Scripts]]
- [[_COMMUNITY_Canonical Schema Guardrails|Canonical Schema Guardrails]]
- [[_COMMUNITY_Leave-One-CSV Validation|Leave-One-CSV Validation]]
- [[_COMMUNITY_QRDQN Training|QRDQN Training]]
- [[_COMMUNITY_Random Forest Baseline|Random Forest Baseline]]
- [[_COMMUNITY_NSL-KDD Loader|NSL-KDD Loader]]
- [[_COMMUNITY_Lab Traffic Generator|Lab Traffic Generator]]

## God Nodes (most connected - your core abstractions)
1. `RLDatasetDefenderEnv` - 24 edges
2. `CICIDSLoadConfig` - 22 edges
3. `Historical NSL-KDD Branch` - 21 edges
4. `NSL-KDD Dataset` - 20 edges
5. `Phase 2 Offline Inference` - 12 edges
6. `Robust v2 Inference Pipeline` - 10 edges
7. `Canonical Flow Schema` - 9 edges
8. `CICIDS2017 Dataset` - 9 edges
9. `_prepare_cicids_features()` - 8 edges
10. `ProgressCallback` - 8 edges

## Surprising Connections (you probably didn't know these)
- `CICIDS2017 Dataset` --semantically_similar_to--> `NSL-KDD Dataset`  [INFERRED] [semantically similar]
  README.md → experiments\nslkdd_experiments.md
- `Entrena un RandomForestClassifier sobre NSL-KDD y devuelve el modelo.` --uses--> `CICIDSLoadConfig`  [INFERRED]
  src\baseline_random_forest.py → src\load_cicids2017.py
- `Evalúa el Random Forest y muestra matriz de confusión + informe de clasificación` --uses--> `CICIDSLoadConfig`  [INFERRED]
  src\baseline_random_forest.py → src\load_cicids2017.py
- `train_rl_defender.py — Entrenamiento de agente defensor RL sobre CICIDS2017.` --uses--> `CICIDSLoadConfig`  [INFERRED]
  src\train_rl_defender.py → src\load_cicids2017.py
- `Devuelve una función creadora de entornos para DummyVecEnv.` --uses--> `CICIDSLoadConfig`  [INFERRED]
  src\train_rl_defender.py → src\load_cicids2017.py

## Hyperedges (group relationships)
- **Canonical Observation Pipeline** — concept_canonical_flow_schema, concept_missingness_mask, concept_observation_vector_152, concept_cicids2017_dataset [EXTRACTED 1.00]
- **Phase 2 Lab Workflow** — concept_phase_2_offline_inference, concept_private_lab_workflow, concept_robust_v2_inference_pipeline, concept_domain_shift_diagnostics [EXTRACTED 1.00]
- **Historical NSL-KDD Track** — concept_historical_nsl_kdd_branch, concept_nsl_kdd_dataset, concept_random_forest_baseline, concept_reward_shaping [EXTRACTED 1.00]

## Communities

### Community 0 - "Environment and Tuning"
Cohesion: 0.09
Nodes (22): BaseCallback, Calcula la recompensa en función de la etiqueta real, la acción         y la co, Entorno RL para un defensor que decide PERMIT/BLOCK sobre muestras etiquetadas., RLDatasetDefenderEnv, _evaluate_f1(), main(), objective(), parse_args() (+14 more)

### Community 1 - "Project Docs and Phase Plan"
Cohesion: 0.14
Nodes (18): Active Blocking Future Work, Artifact-Backed Results, CICIDS2017 Dataset, Documentation Map, Domain Shift Diagnostics, Leave-One-Exact-CSV-Out Validation, Phase 1 Offline Training, Phase 2 Offline Inference (+10 more)

### Community 2 - "CICIDS2017 Loader"
Cohesion: 0.14
Nodes (26): CICIDSLoadConfig, _clean_rows(), _coerce_numeric_features(), _drop_identifier_like_columns(), _find_label_column(), list_cicids2017_csv_files(), _list_csv_files(), _load_all_csvs() (+18 more)

### Community 3 - "NSL-KDD History and Assets"
Cohesion: 0.24
Nodes (5): RLDatasetDefenderEnv, Historical NSL-KDD Branch, NSL-KDD Dataset, Random Forest Baseline, Reward Shaping

### Community 4 - "Phase 2 Inference Scripts"
Cohesion: 0.16
Nodes (17): batched_predict(), load_model(), main(), maybe_convert_time_units(), batched_predict(), compute_diagnostics(), compute_truth_metrics(), load_model() (+9 more)

### Community 5 - "Canonical Schema Guardrails"
Cohesion: 0.19
Nodes (13): CanonicalResult, get_canonical_feature_names(), get_observation_feature_names(), map_to_canonical(), canonical_schema.py — Definición formal del esquema canónico de features (FEATUR, Resultado de mapear un DataFrame al esquema canónico., Mapea un DataFrame al esquema canónico de features.      Parameters     -----, Devuelve la lista de nombres de features canónicas (sin máscara). (+5 more)

### Community 6 - "Leave-One-CSV Validation"
Cohesion: 0.25
Nodes (13): _build_aggregate_results(), _compute_reward_total(), evaluate_model_direct(), main(), make_env_fn(), _metrics_from_confusion(), parse_args(), validate_leave_one_csv_out.py — Validación leave-one-exact-CSV-out para CICIDS20 (+5 more)

### Community 7 - "QRDQN Training"
Cohesion: 0.36
Nodes (7): evaluate_model(), main(), make_env_fn(), parse_args(), train_rl_defender.py — Entrenamiento de agente defensor RL sobre CICIDS2017., Devuelve una función creadora de entornos para DummyVecEnv., Evalúa el agente sobre test set.     Devuelve dict con métricas clave.

### Community 8 - "Random Forest Baseline"
Cohesion: 0.47
Nodes (5): evaluate_random_forest(), main(), Entrena un RandomForestClassifier sobre NSL-KDD y devuelve el modelo., Evalúa el Random Forest y muestra matriz de confusión + informe de clasificación, train_random_forest()

### Community 9 - "NSL-KDD Loader"
Cohesion: 0.6
Nodes (4): _download_nsl_kdd_via_kagglehub(), _ensure_dataset_local_dir(), load_nsl_kdd_binary(), Carga NSL-KDD desde Kaggle (hassan06/nslkdd), lo preprocesa y devuelve:

### Community 10 - "Lab Traffic Generator"
Cohesion: 0.83
Nodes (3): http_get(), main(), tcp_connect()

## Knowledge Gaps
- **26 isolated node(s):** `predict_real_traffic_v2.py — Robust Phase 2 offline inference pipeline.  Loads`, `If time columns look like seconds (median < 1), convert to microseconds.`, `Load QRDQN model, falling back to DQN if sb3_contrib is unavailable.`, `Run model.predict in batches to avoid OOM on large flow CSVs.`, `Compute z-score diagnostics on scaled features (first _N_CANON dims only).` (+21 more)
  These have ≤1 connection - possible missing edges or undocumented components.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Canonical Flow Schema` connect `Canonical Schema Guardrails` to `Project Docs and Phase Plan`, `CICIDS2017 Loader`, `NSL-KDD History and Assets`?**
  _High betweenness centrality (0.221) - this node is a cross-community bridge._
- **Why does `CICIDS2017 Dataset` connect `Project Docs and Phase Plan` to `Environment and Tuning`, `CICIDS2017 Loader`, `NSL-KDD History and Assets`, `Canonical Schema Guardrails`, `Leave-One-CSV Validation`, `QRDQN Training`, `Random Forest Baseline`?**
  _High betweenness centrality (0.209) - this node is a cross-community bridge._
- **Why does `CICIDSLoadConfig` connect `CICIDS2017 Loader` to `Random Forest Baseline`, `Environment and Tuning`, `Leave-One-CSV Validation`, `QRDQN Training`?**
  _High betweenness centrality (0.164) - this node is a cross-community bridge._
- **Are the 15 inferred relationships involving `RLDatasetDefenderEnv` (e.g. with `train_rl_defender.py — Entrenamiento de agente defensor RL sobre CICIDS2017.` and `Devuelve una función creadora de entornos para DummyVecEnv.`) actually correct?**
  _`RLDatasetDefenderEnv` has 15 INFERRED edges - model-reasoned connections that need verification._
- **Are the 17 inferred relationships involving `CICIDSLoadConfig` (e.g. with `Entrena un RandomForestClassifier sobre NSL-KDD y devuelve el modelo.` and `Evalúa el Random Forest y muestra matriz de confusión + informe de clasificación`) actually correct?**
  _`CICIDSLoadConfig` has 17 INFERRED edges - model-reasoned connections that need verification._
- **Are the 13 inferred relationships involving `Historical NSL-KDD Branch` (e.g. with `KDDTest+.txt` and `KDDTest-21.txt`) actually correct?**
  _`Historical NSL-KDD Branch` has 13 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `NSL-KDD Dataset` (e.g. with `Canonical Flow Schema` and `CICIDS2017 Dataset`) actually correct?**
  _`NSL-KDD Dataset` has 2 INFERRED edges - model-reasoned connections that need verification._