# Graph Report - C:\Users\Rivero\Desktop\TFG_CYBER_AI  (2026-04-25)

## Corpus Check
- Corpus is ~19,402 words - fits in a single context window. You may not need a graph.

## Summary
- 187 nodes · 339 edges · 8 communities detected
- Extraction: 82% EXTRACTED · 18% INFERRED · 0% AMBIGUOUS · INFERRED: 60 edges (avg confidence: 0.77)
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

## God Nodes (most connected - your core abstractions)
1. `RLDatasetDefenderEnv` - 27 edges
2. `CICIDSLoadConfig` - 26 edges
3. `main()` - 11 edges
4. `main()` - 11 edges
5. `decide_and_run()` - 10 edges
6. `_prepare_cicids_features()` - 9 edges
7. `load_cicids2017_binary()` - 9 edges
8. `load_cicids2017_split()` - 9 edges
9. `map_to_canonical()` - 8 edges
10. `ProgressCallback` - 8 edges

## Surprising Connections (you probably didn't know these)
- `Dataset Adapter Contract` --references--> `load_cicids2017_split()`  [EXTRACTED]
  .github/AGENT_CONTEXT.md → src/load_cicids2017.py
- `CICIDS2017 Primary Dataset` --references--> `CICIDSLoadConfig`  [EXTRACTED]
  .github/AGENT_CONTEXT.md → src/load_cicids2017.py
- `Check C Hard CSV/Day Split` --references--> `check_c_csv_split()`  [EXTRACTED]
  docs/results.md → src/validate_checks.py
- `main()` --calls--> `parse_args()`  [INFERRED]
  scripts/graphify_auto_update.py → src/validate_leave_one_csv_out.py
- `Robust v2 Inference Pipeline` --references--> `compute_diagnostics()`  [EXTRACTED]
  docs/AGENT_CONTEXT.md → scripts/predict_real_traffic_v2.py

## Hyperedges (group relationships)
- **Canonical Observation Pipeline** — concept_canonical_flow_schema, concept_76_canonical_features, concept_missingness_mask, concept_observation_vector_152, src_canonical_schema_py [EXTRACTED 1.00]
- **CICIDS2017 Training and Validation Pipeline** — concept_cicids2017_dataset, concept_qrdqn_training_pipeline, concept_validation_suite, src_load_cicids2017_py, src_train_rl_defender_py, src_validate_checks_py [EXTRACTED 1.00]
- **Phase 2 Lab Inference Workflow** — concept_phase2_offline_inference, concept_private_lab_workflow, concept_robust_v2_inference_pipeline, concept_domain_shift_risk, scripts_predict_real_traffic_v2_py [EXTRACTED 1.00]
- **Documentation and Defense Alignment** — doc_docs_readme, doc_results, doc_defensa_progreso, doc_defensa_script, concept_artifact_backed_results, concept_defense_positioning [EXTRACTED 1.00]

## Communities

### Community 0 - "Community 0"
Cohesion: 0.11
Nodes (32): evaluate_random_forest(), main(), Entrena un RandomForestClassifier sobre NSL-KDD y devuelve el modelo., Evalúa el Random Forest y muestra matriz de confusión + informe de clasificación, train_random_forest(), Anti-Leakage Policy, CICIDSLoadConfig, _clean_rows() (+24 more)

### Community 1 - "Community 1"
Cohesion: 0.08
Nodes (30): Active Blocking Future Work, Honest Defense Positioning, Domain Shift Risk, Two-VM Private Lab Topology, Phase 2 Offline Inference, Private Lab Workflow, Reproducible Run Artifacts, Robust v2 Inference Pipeline (+22 more)

### Community 2 - "Community 2"
Cohesion: 0.11
Nodes (21): CICIDS2017 Primary Dataset, QRDQN Training Pipeline, Current Reward Configuration, Python Runtime Dependencies, Calcula la recompensa en función de la etiqueta real, la acción         y la co, Entorno RL para un defensor que decide PERMIT/BLOCK sobre muestras etiquetadas., RLDatasetDefenderEnv, evaluate_model() (+13 more)

### Community 3 - "Community 3"
Cohesion: 0.14
Nodes (18): CanonicalResult, get_canonical_feature_names(), get_observation_feature_names(), map_to_canonical(), canonical_schema.py — Definición formal del esquema canónico de features (FEATUR, Resultado de mapear un DataFrame al esquema canónico., Mapea un DataFrame al esquema canónico de features.      Parameters     -----, Devuelve la lista de nombres de features canónicas (sin máscara). (+10 more)

### Community 4 - "Community 4"
Cohesion: 0.15
Nodes (16): BaseCallback, Check A Direct Evaluation, Check B Shuffled-Label Anti-Leakage, Check C Hard CSV/Day Split, Validation Suite, check_a_direct_eval(), check_b_shuffled_labels(), check_c_csv_split() (+8 more)

### Community 5 - "Community 5"
Cohesion: 0.24
Nodes (14): decide_and_run(), _diff_range(), _has_structural_patch(), _is_code(), _is_ignored(), _is_semantic_source(), main(), _normalize() (+6 more)

### Community 6 - "Community 6"
Cohesion: 0.12
Nodes (17): Artifact-Backed Results, Binary PERMIT/BLOCK Defender Actions, C03 Best Historical CICIDS2017 Run, Graphify Corpus Policy, NSL-KDD Historical Benchmark, Phase 1 Offline Training and Validation, RL Cyber Defender Project, AGENTS.md Project Operating Rules (+9 more)

### Community 7 - "Community 7"
Cohesion: 0.24
Nodes (14): Leave-One-Exact-CSV-Out Validation, _build_aggregate_results(), _compute_reward_total(), evaluate_model_direct(), main(), make_env_fn(), _metrics_from_confusion(), parse_args() (+6 more)

## Knowledge Gaps
- **36 isolated node(s):** `predict_real_traffic_v2.py — Robust Phase 2 offline inference pipeline.  Loads`, `If time columns look like seconds (median < 1), convert to microseconds.`, `Load QRDQN model, falling back to DQN if sb3_contrib is unavailable.`, `Run model.predict in batches to avoid OOM on large flow CSVs.`, `Compute z-score diagnostics on scaled features (first _N_CANON dims only).` (+31 more)
  These have ≤1 connection - possible missing edges or undocumented components.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `CICIDSLoadConfig` connect `Community 0` to `Community 2`, `Community 4`, `Community 7`?**
  _High betweenness centrality (0.338) - this node is a cross-community bridge._
- **Why does `map_to_canonical()` connect `Community 3` to `Community 0`, `Community 1`, `Community 6`?**
  _High betweenness centrality (0.260) - this node is a cross-community bridge._
- **Why does `Canonical Flow Schema` connect `Community 3` to `Community 0`, `Community 2`, `Community 6`?**
  _High betweenness centrality (0.232) - this node is a cross-community bridge._
- **Are the 17 inferred relationships involving `RLDatasetDefenderEnv` (e.g. with `train_rl_defender.py — Entrenamiento de agente defensor RL sobre CICIDS2017.` and `Devuelve una función creadora de entornos para DummyVecEnv.`) actually correct?**
  _`RLDatasetDefenderEnv` has 17 INFERRED edges - model-reasoned connections that need verification._
- **Are the 20 inferred relationships involving `CICIDSLoadConfig` (e.g. with `Entrena un RandomForestClassifier sobre NSL-KDD y devuelve el modelo.` and `Evalúa el Random Forest y muestra matriz de confusión + informe de clasificación`) actually correct?**
  _`CICIDSLoadConfig` has 20 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `main()` (e.g. with `map_to_canonical()` and `apply_percentile_clipping()`) actually correct?**
  _`main()` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `main()` (e.g. with `list_cicids2017_csv_files()` and `CICIDSLoadConfig`) actually correct?**
  _`main()` has 4 INFERRED edges - model-reasoned connections that need verification._