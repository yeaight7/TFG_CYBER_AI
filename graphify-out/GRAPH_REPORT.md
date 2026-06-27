# Graph Report - TFG_CYBER_AI  (2026-06-27)

## Corpus Check
- 90 files · ~179,581 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1448 nodes · 1528 edges · 95 communities (83 shown, 12 thin omitted)
- Extraction: 95% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 65 edges (avg confidence: 0.78)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `c8372ce7`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

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
- [[_COMMUNITY_Community 61|Community 61]]
- [[_COMMUNITY_Community 62|Community 62]]
- [[_COMMUNITY_Community 63|Community 63]]
- [[_COMMUNITY_Community 64|Community 64]]
- [[_COMMUNITY_Community 65|Community 65]]
- [[_COMMUNITY_Community 66|Community 66]]
- [[_COMMUNITY_Community 67|Community 67]]
- [[_COMMUNITY_Community 68|Community 68]]
- [[_COMMUNITY_Community 69|Community 69]]
- [[_COMMUNITY_Community 70|Community 70]]
- [[_COMMUNITY_Community 71|Community 71]]
- [[_COMMUNITY_Community 72|Community 72]]
- [[_COMMUNITY_Community 73|Community 73]]
- [[_COMMUNITY_Community 74|Community 74]]
- [[_COMMUNITY_Community 75|Community 75]]
- [[_COMMUNITY_Community 76|Community 76]]
- [[_COMMUNITY_Community 77|Community 77]]
- [[_COMMUNITY_Community 78|Community 78]]
- [[_COMMUNITY_Community 79|Community 79]]
- [[_COMMUNITY_Community 80|Community 80]]
- [[_COMMUNITY_Community 81|Community 81]]
- [[_COMMUNITY_Community 82|Community 82]]
- [[_COMMUNITY_Community 83|Community 83]]
- [[_COMMUNITY_Community 84|Community 84]]
- [[_COMMUNITY_Community 85|Community 85]]
- [[_COMMUNITY_Community 87|Community 87]]
- [[_COMMUNITY_Community 88|Community 88]]
- [[_COMMUNITY_Community 89|Community 89]]
- [[_COMMUNITY_Community 90|Community 90]]
- [[_COMMUNITY_Community 91|Community 91]]
- [[_COMMUNITY_Community 92|Community 92]]
- [[_COMMUNITY_Community 93|Community 93]]
- [[_COMMUNITY_Community 94|Community 94]]

## God Nodes (most connected - your core abstractions)
1. `Defensa TFG — Guion oral actualizado` - 20 edges
2. `Investigación complementaria para defensa: componentes restantes del proyecto` - 20 edges
3. `Investigación técnica (Defensa TFG): modelos adicionales, hiperparámetros y validación` - 17 edges
4. `RLDatasetDefenderEnv` - 15 edges
5. `Defensa TFG — Progreso, mensajes clave y hechos verificables` - 15 edges
6. `State of the Art` - 14 edges
7. `Safe Claims` - 14 edges
8. `Literature Matrix` - 14 edges
9. `Repository Audit — TFG_CYBER_AI` - 13 edges
10. `Guía de implementación — limpieza y realineamiento del repo TFG_CYBER_AI` - 13 edges

## Surprising Connections (you probably didn't know these)
- `test_compute_diagnostics()` --calls--> `compute_diagnostics()`  [INFERRED]
  tests/test_predict_real_traffic_v2.py → scripts/predict_real_traffic_v2.py
- `main()` --calls--> `map_to_canonical()`  [INFERRED]
  scripts/predict_real_traffic_v2.py → src/canonical_schema.py
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

## Communities (95 total, 12 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.06
Nodes (59): _dup_stats(), main(), _pct(), analyze_duplicates.py — Phase-0 / Task A1 (read-only analysis).  Quantifies, WIT, View a 2-D float32 array as a 1-D array of byte-exact per-row records., _void_rows(), fail(), main() (+51 more)

### Community 1 - "Community 1"
Cohesion: 0.04
Nodes (42): Canonical Flow Schema, Phase 2 Offline Inference, MAIN_qrdqn_cicids2017_canonical_full_random_20260609_193655, Data Structure and Canonical Schema Research, Phase 2 Agent Context, Documentation Index, Audits Index, code:text (runs/cicids2017/MAIN_qrdqn_cicids2017_canonical_full_random_) (+34 more)

### Community 2 - "Community 2"
Cohesion: 0.04
Nodes (44): 1.1 Benchmark NIDS datasets and flow feature tools, 1.2 Flow-based NIDS and ML/DL methods, 1.3 Reinforcement learning fundamentals and extensions (DQN, Double/Dueling, distributional, QRDQN), 1.4 RL for NIDS and autonomous cyber defence, 1.5 Evaluation methodology, cross-dataset generalization, and external validation, 1.6 Cost-sensitive IDS, FP/FN trade-offs, and reward design, 1.7 Reproducibility, tools, and dataset-as-environment formulations, 1.8 Methodological risks: adversarial robustness and over-optimistic metrics (+36 more)

### Community 3 - "Community 3"
Cohesion: 0.05
Nodes (40): 10. Posicionamiento honesto del trabajo, 11. Mensajes fuertes para el tribunal, 12. Riesgos discursivos que conviene evitar, 13. Cierre recomendado, 1. Mensaje central del TFG, 2. Estado actual del proyecto, 3. Invariantes técnicos que debes mantener en la exposición, 4. Datasets y papel de cada uno (+32 more)

### Community 4 - "Community 4"
Cohesion: 0.07
Nodes (32): apply_percentile_clipping(), apply_z_clipping(), scaling_utils.py — Clipping utilities for outlier handling in RL inference pipel, Clamp each feature to its training percentile range [p_low, p_high].      Appl, Clamp scaled features to [-max_z, +max_z].      Applied to scaled features **a, batched_predict(), compute_diagnostics(), compute_truth_metrics() (+24 more)

### Community 5 - "Community 5"
Cohesion: 0.06
Nodes (32): 1. Create the Network, 2. Create Firewall Rules, 3. Create the VMs, Adapting to Other Providers, Attacker Setup, Benign, Capture and Feature Extraction, code:bash (gcloud compute networks create tfg-lab-vpc \) (+24 more)

### Community 6 - "Community 6"
Cohesion: 0.06
Nodes (31): 0. Cómo usar esta guía, 10. Reorg de `docs/` ya ejecutada (2026-06-25), 11. Carry-forward de auditorías previas (consolidado 2026-06-25), 1.1 Verdad canónica de hiperparámetros (fuente: `src/train_rl_defender.py` + `MAIN/config.json`), 1. Marco oficial del proyecto (decidido por el autor, 2026-06-25), 2. Decisiones tomadas (2026-06-25), 3. Hallazgos priorizados, 4. Buckets de acción (+23 more)

### Community 7 - "Community 7"
Cohesion: 0.06
Nodes (30): Actor–critic, policy-gradient, and SAC-based IDS, Autoencoders and unsupervised/semi-supervised DL, Classical supervised models, CNNs, RNNs, and hybrid CNN–LSTM models, code:bibtex (@article{Survey2023MLDLIDS,), code:bibtex (@article{Survey2025DLNIDS,), code:bibtex (@article{Yang2026DRLNIDSSurvey,), code:bibtex (@article{HypergraphNIDS2024,) (+22 more)

### Community 8 - "Community 8"
Cohesion: 0.07
Nodes (29): Context, Decisions locked (from owner, 2026-06-27), Decisions locked — round 2 (owner, 2026-06-27) — RESOLVED execution gates, Decisions locked — round 3 (owner, 2026-06-27) — post-A1, Detailed tasks, Global verification, How to use this tracker, Master tracker (+21 more)

### Community 9 - "Community 9"
Cohesion: 0.07
Nodes (29): 1.1 Flow‑based NIDS and classical NIDS, 1.2 Public cybersecurity datasets (flow‑based and classic), 1.3 ML / DL for NIDS (supervised), 1.4 DRL for intrusion detection (dataset‑as‑environment), 1.5 RL‑based autonomous / adaptive cyber defense, 1.6 Evaluation methodology and dataset quality, 1. Executive source map, 2. Antecedentes y conceptos básicos (+21 more)

### Community 10 - "Community 10"
Cohesion: 0.07
Nodes (28): Artifact Locations, Baseline Metrics, C03 (best historical probe), Check A — Direct Evaluation, Check B — Shuffled Labels, Check C — Hard CSV/Day Split, CICIDS2017 Training Runs, code:python (REWARD_CONFIG = {) (+20 more)

### Community 11 - "Community 11"
Cohesion: 0.07
Nodes (27): 10) HCLR‑IDS 2025 \& DRL‑IDS SDN 2025 (P13/P14)[^11][^12], 1) López‑Martín 2021 – Extended RBFNN with offline RL (P2/P4)[^13][^4], 1. Short verdict, 2) López‑Martín 2020 – DRL for supervised intrusion detection (P1/P3)[^5][^1], 2. Paper matrix, 3. Closest works to my thesis, 3) Ren 2022 – ID‑RDRL on CSE‑CIC‑IDS2018 (P5/P11)[^3][^2], 4. Reward design patterns (+19 more)

### Community 12 - "Community 12"
Cohesion: 0.07
Nodes (26): 10. Algoritmo principal, 11. Resultados principales en CICIDS2017, 12. Validación rigurosa, 13. Nueva validación leave-one-exact-CSV-out, 14. Fase 2: tráfico real en laboratorio, 15. Qué ocurrió en la Fase 2, 16. Qué aporta esto científicamente, 17. Limitaciones (+18 more)

### Community 13 - "Community 13"
Cohesion: 0.07
Nodes (25): 1. Prepare the Private Lab, 2. Generate Labelled Traffic, 3. Capture Traffic, 4. Extract Flow Features, 5. Map to the Canonical Schema, 6. Run Robust Offline Inference, 7. Store Run Artifacts, 8. Review the Results (+17 more)

### Community 14 - "Community 14"
Cohesion: 0.14
Nodes (24): collect_environment_metadata(), configure_torch_runtime(), evaluate_model(), main(), make_env_fn(), _package_version(), parse_args(), train_rl_defender.py — Entrenamiento de agente defensor RL sobre CICIDS2017. (+16 more)

### Community 15 - "Community 15"
Cohesion: 0.11
Nodes (14): Calcula la recompensa en función de la etiqueta real, la acción         y la co, Entorno RL para un defensor que decide PERMIT/BLOCK sobre muestras etiquetadas., RLDatasetDefenderEnv, _evaluate_f1(), main(), objective(), parse_args(), tune_hparams.py -- Optimizacion de hiperparametros con Optuna para QRDQN sobre C (+6 more)

### Community 16 - "Community 16"
Cohesion: 0.08
Nodes (24): 1. Conceptual foundation, 2. Source matrix, 3.1 Simple correct/incorrect rewards, 3.2 Class-weighted rewards, 3.3 False-negative-heavy (cost-sensitive) rewards, 3.4 Precision- / recall-oriented rewards, 3.5 Dynamic/adaptive rewards, 3.6 Risk-based/system-cost rewards (+16 more)

### Community 17 - "Community 17"
Cohesion: 0.08
Nodes (24): 1. Executive summary, 2. Dataset comparison matrix, 3. CICIDS2017 deep dive, 4. Leakage and evaluation risk matrix, 5.1 Public dataset: CICIDS2017 (train/validation/internal test), 5.2 External validation with private lab traffic, 5.3 Fallback plan (if lab traffic is limited), 5. Recommended evaluation protocol for my thesis (+16 more)

### Community 18 - "Community 18"
Cohesion: 0.08
Nodes (23): Claim 10: QR-DQN is justified as an experimental distributional RL choice, Claim 11: Cost-sensitive rewards match IDS risk structure, Claim 12: Phase 2 is offline inference, not active blocking, Claim 13: Phase 2 behavior is artifact-specific, Claim 1: Flow-based NIDS is an established representation, Claim 2: CICIDS2017 is a reasonable primary benchmark, Claim 3: The project uses a fixed canonical flow schema, Claim 4: Supervised baselines are necessary (+15 more)

### Community 19 - "Community 19"
Cohesion: 0.09
Nodes (22): Candidate paragraphs for integration, Claims that are safe vs unsafe, code:bibtex (@article{Yang2026DRLNIDSurvey,), code:bibtex (@article{Kheddar2024RLIDSReview,), code:bibtex (@article{Javadpour2026BeyondRLNetworkSecurity,), code:bibtex (@article{Standen2021CybORG,), code:bibtex (@article{WatkinsDayan1992QLearning,), code:bibtex (@article{Lin1992ReactiveAgents,) (+14 more)

### Community 20 - "Community 20"
Cohesion: 0.12
Nodes (17): batched_predict(), load_model(), main(), maybe_convert_time_units(), CanonicalResult, get_canonical_feature_names(), get_observation_feature_names(), map_to_canonical() (+9 more)

### Community 21 - "Community 21"
Cohesion: 0.09
Nodes (21): 10. Security / privacy audit, 11. Prioritized action plan, 12. Questions for the owner (only the genuinely blocking ones), 1. Executive summary, 2. What the repository currently implements, 3. Strong points, 4. Critical issues, 5. Documentation / report alignment issues (+13 more)

### Community 22 - "Community 22"
Cohesion: 0.1
Nodes (20): 10) Conclusión de esta investigación complementaria, 1) Mapa de “qué es cada cosa” (resto del proyecto), 2.1 Random Forest baseline, 2.2 Optuna (no es modelo, es algoritmo de búsqueda), 2) Algoritmos adicionales (aparte de QRDQN), 3.1 Validación leave-one-exact-CSV-out, 3.2 Check B y Check C (detalles prácticos no obvios), 3.3 Pipeline robusto de inferencia: utilidades de clipping (+12 more)

### Community 23 - "Community 23"
Cohesion: 0.1
Nodes (20): Afirmaciones a evitar, Afirmaciones seguras, Afirmaciones seguras, afirmaciones a evitar y glosario, Base de fuentes, Bloque BibTeX esencial, Brecha que tu TFG puede reclamar sin exagerar, Cómo puede responder tu diseño experimental, code:bibtex (@techreport{ScarfoneMell2007,) (+12 more)

### Community 24 - "Community 24"
Cohesion: 0.1
Nodes (20): Claims to Avoid, code:bibtex (@techreport{ScarfoneMell2007,), Essential BibTeX Block, Essential Selection, Glossary of Terms, How your experimental design can answer, Methodological Critique of the Area, Narrative Synthesis (+12 more)

### Community 25 - "Community 25"
Cohesion: 0.1
Nodes (20): 1. Core Thesis Research Areas, 2. Source Clusters, 3. Must-Cite Sources, 4. Useful but Secondary Sources, 5. Weak / Risky / Suspicious Sources, 6. Missing Research Areas, 7. Recommended Next Research Prompts, Autonomous Cyber Defense (+12 more)

### Community 26 - "Community 26"
Cohesion: 0.1
Nodes (19): code:text (TFG_CYBER_AI/), code:bash (pip install -r requirements.txt), code:bash (pip install -r requirements-runpod-cu130.txt), code:bash (python src/train_rl_defender.py --smoke), code:bash (python src/validate_checks.py --model models/<MODEL>.zip --c), code:bash (python src/validate_leave_one_csv_out.py --timesteps 30000), code:bash (python scripts/predict_real_traffic_v2.py \), Core Technical Invariants (+11 more)

### Community 27 - "Community 27"
Cohesion: 0.11
Nodes (18): Check C Detail, CICIDS2017 + QRDQN Experiment History, code:python (REWARD_CONFIG = {), code:bash (# 1M train rows (same fixed test partition as the main run):), Commands, Historical Interpretation, How To Read This Page, Leave-One-Exact-CSV-Out (+10 more)

### Community 28 - "Community 28"
Cohesion: 0.11
Nodes (17): 1) Alcance, 2.1 Baseline clásico: Random Forest, 2.2 DQN como fallback operativo, 2.3 Búsqueda de hiperparámetros con Optuna, 2) Modelos y algoritmos presentes además del núcleo QRDQN, 3) Arquitecturas de red usadas en el proyecto (sin entrar en detalle de QRDQN), 4.1 Entrenamiento principal — experimento oficial (perfil `main-experiment`, run MAIN), 4.2 Leave-one-exact-CSV-out (`src/validate_leave_one_csv_out.py`) (+9 more)

### Community 29 - "Community 29"
Cohesion: 0.11
Nodes (17): Arquitectura de red y parámetros de la API, code:block1 (flowchart TD), code:python (# Pseudocódigo fiel al entrenamiento de SB3-Contrib QRDQN), code:block3 (flowchart LR), code:python (# Experimento oficial: perfil `main-experiment` (run MAIN)), code:text (Acción PERMIT:), code:python (import torch), Consideraciones prácticas, evaluación y reproducibilidad (+9 more)

### Community 30 - "Community 30"
Cohesion: 0.11
Nodes (17): 1. Files Modified in This Session, 2. Best Material Produced, 3. Material That Still Needs Human Review, 4. Missing Citations and Verification Points, 5. Suggested Next Prompt — Codex / Claude: Integrate SoA into Draft, 6. Suggested Next Prompt — Perplexity / Deep Research: Verify Citations, 7. Git Commands to Inspect Changes, 8. State of All Nightly Files (+9 more)

### Community 31 - "Community 31"
Cohesion: 0.12
Nodes (16): 1) Alcance de este documento, 2) Arquitectura general del proyecto, 3.1 Configuración de carga, 3.2 Limpieza y saneado, 3.3 Imputación y máscara de missingness, 3) Procesado de datos (CICIDS2017) y anti-leakage, 4.1 Interfaces Gymnasium, 4.2 Recompensa (+8 more)

### Community 32 - "Community 32"
Cohesion: 0.12
Nodes (16): 07 — Artefactos, scripts, tests y validacion, 1) Pipeline conceptual: donde se aprende y donde no, 2) Artefactos de entrenamiento, 3) Metricas y como leerlas, 4) Clipping, z-scores y domain shift, 5) Mapa de scripts principales, 6) Tests unitarios vs validaciones experimentales, 7) Que validamos, como y por que (+8 more)

### Community 33 - "Community 33"
Cohesion: 0.12
Nodes (16): 1.1 "Why use RL instead of a supervised classifier?", 1.2 "Is CICIDS2017 still representative of modern traffic?", 1.3 "Why QRDQN specifically? What is the justification over standard DQN?", 1.4 "The dataset-as-environment is just classification with an RL wrapper. Why is it RL?", 1.5 "Can PERMIT/BLOCK decisions be implemented in a real network?", 1.6 "Your external validation is benign-only — that is not real external validation.", 1.7 "Why those specific training sizes (100k / 250k / 500k / 1M / 2M)?", 1. Core Examiner Challenges (+8 more)

### Community 34 - "Community 34"
Cohesion: 0.21
Nodes (17): Canonical Feature Schema, CICIDS2017 Limitations, Cost-Sensitive Reward, Data Efficiency Experiments, Dataset as Environment, Distributional RL (QRDQN), Evaluation Leakage, Supervised Baselines (Random Forest) (+9 more)

### Community 35 - "Community 35"
Cohesion: 0.12
Nodes (15): 02 — Datos, esquema canónico y preprocesado, 1) Datasets y su papel, 2) Decisión arquitectónica clave: esquema canónico, 3) Máscara de missingness (semántica), 4) Política anti-leakage, 5) Preprocesado en CICIDS2017, 6) Modos de split y por qué importan, 7) Puntos finos que conviene dominar (+7 more)

### Community 36 - "Community 36"
Cohesion: 0.12
Nodes (16): code:bibtex (@article{MaliciousBehavior2021,), code:bibtex (@misc{Stamus2024IDSTypes,), code:bibtex (@misc{Varonis2023FlowMonitoring,), code:bibtex (@misc{Faddom2023NetFlowIPFIX,), code:bibtex (@article{ContextNetFlow2026,), code:bibtex (@article{FlowTutorial2025,), code:bibtex (@article{TemporalNetFlow2025,), code:bibtex (@article{DeploymentFramework2026,) (+8 more)

### Community 37 - "Community 37"
Cohesion: 0.18
Nodes (12): BaseCallback, check_a_direct_eval(), check_b_shuffled_labels(), check_c_csv_split(), main(), parse_args(), ProgressCallback, validate_checks.py — Validación de resultados experimentales del agente RL.  I (+4 more)

### Community 38 - "Community 38"
Cohesion: 0.13
Nodes (14): 10. Methodological Pitfalls in ML-Based NIDS, 11. External Validation and Lab-Captured Traffic, 12. Positioning of This Thesis, 13. Data Efficiency and Training-Scale Evaluation, 1. Network Intrusion Detection Systems, 2. Flow-Based Traffic Representation, 3. Public Datasets for Network Intrusion Detection, 4. CICIDS2017 as the Main Internal Benchmark (+6 more)

### Community 39 - "Community 39"
Cohesion: 0.13
Nodes (14): 1. Conceptual explanation, 2. Source matrix, 3. Arguments in favor, 4. Arguments against, 5. How to defend this formulation in your thesis, 6. Suggested subsection, 7. Codex handoff, Classification-as-RL Dossier (+6 more)

### Community 40 - "Community 40"
Cohesion: 0.13
Nodes (14): Cluster 10 — Data Efficiency, Cluster 11 — Repository Artifacts, Cluster 1 — Flow-Based Traffic Representation, Cluster 2 — Public NIDS Datasets, Cluster 3 — CICIDS2017 Quality Concerns, Cluster 4 — Supervised ML and DL for NIDS, Cluster 5 — Reinforcement Learning Foundations, Cluster 6 — RL and DRL for Intrusion Detection (+6 more)

### Community 41 - "Community 41"
Cohesion: 0.25
Nodes (13): _build_aggregate_results(), _compute_reward_total(), evaluate_model_direct(), main(), make_env_fn(), _metrics_from_confusion(), parse_args(), validate_leave_one_csv_out.py — Validación leave-one-exact-CSV-out para CICIDS20 (+5 more)

### Community 42 - "Community 42"
Cohesion: 0.14
Nodes (13): 1. DQN foundation, 2. From DQN to Distributional RL, 3.1 Quantiles and return distribution, 3.2 Quantile regression and quantile Huber loss, 3.3 Action selection and difference from DQN, 3. QRDQN explanation, 4. Source matrix, 5. QRDQN in cybersecurity (+5 more)

### Community 43 - "Community 43"
Cohesion: 0.14
Nodes (13): Attack-Family Error Analysis, Data-Efficiency Curve, External Lab Validation, False Positive / False Negative Analysis, Internal Benchmark Design, Leakage Controls, Methodology Handoff, Methodology Writing Checklist (+5 more)

### Community 44 - "Community 44"
Cohesion: 0.34
Nodes (13): decide_and_run(), _diff_range(), _has_structural_patch(), _is_code(), _is_ignored(), _is_semantic_source(), main(), _normalize() (+5 more)

### Community 45 - "Community 45"
Cohesion: 0.15
Nodes (12): 06 — Glosario esencial + preguntas de tribunal, 1) Glosario mínimo para explicar el proyecto a principiantes, 2) Preguntas frecuentes del tribunal (con respuesta técnica breve), 3) Estructura de explicación en 3 minutos (versión corta), 4) Estructura de explicación en 10–12 minutos (versión defensa), “¿Cómo evitas leakage?”, “¿Cuál es tu principal limitación actual?”, “¿Está lista Phase 2 para producción?” (+4 more)

### Community 46 - "Community 46"
Cohesion: 0.15
Nodes (12): Claims Requiring Verification, code:txt (Audit the citation keys in report/drafts/state_of_the_art.md), code:txt (Build the experimental design chapter from Research/Initial ), Deeper Mathematical Explanation to Add Later, Missing Citations, Overclaim Checks for Next Pass, Places Where the Draft May Be Too Generic, Repository Facts to Cross-Check Later (+4 more)

### Community 47 - "Community 47"
Cohesion: 0.15
Nodes (12): 10. Codex handoff, 1. What CICIDS2017 is, 2. Official facts, 3. Attack categories and labels, 4. Feature extraction and flow representation, 5. Known limitations, 6. Preprocessing checklist for my thesis, 7. Evaluation protocol recommendation (+4 more)

### Community 48 - "Community 48"
Cohesion: 0.15
Nodes (12): 1. Dataset comparison table, 2. Historical evolution, 3.1 Advantages, 3.2 Comparison vs. NSL-KDD and UNSW-NB15, 3.3 Why suitable for flow-level binary PERMIT/BLOCK, 3.4 Why not enough for real-world claims, 3. Why CICIDS2017 is a reasonable main dataset, 4. Dataset choice decision matrix (+4 more)

### Community 49 - "Community 49"
Cohesion: 0.15
Nodes (12): 1. The Literature Landscape, 2. What Is Partially Covered and Leaves Room for Contribution, 3. What Is Weakly Validated in the Literature, 4. Defensible Research Gap, 5. Claims That Must Not Appear in the Thesis, 6. Evidence Mapping, 7. Thesis-Ready Sentences, Research Gap and Thesis Positioning (+4 more)

### Community 50 - "Community 50"
Cohesion: 0.17
Nodes (11): 03 — Entorno RL, algoritmo y entrenamiento, 1) Entorno RL: qué modela exactamente, 2) Recompensa actual por defecto, 3) Sobre el algoritmo principal, 4) Hiperparámetros relevantes en el experimento oficial (perfil `main-experiment`, run MAIN), 5) Presets y timesteps, 6) Artefactos que deja cada entrenamiento, 7) Baseline y tuning (complemento) (+3 more)

### Community 51 - "Community 51"
Cohesion: 0.17
Nodes (11): 05 — Phase 2: laboratorio, inferencia y riesgos abiertos, 1) Qué es exactamente Phase 2 hoy, 2) Flujo operativo resumido, 3) Endurecimiento técnico de la v2, 4) Artefactos esperados de cada run Phase 2, 5) Hallazgo central de Phase 2 (honesto), 6) Laboratorio privado y seguridad operacional, 7) Qué faltaría para pasar a bloqueo activo (+3 more)

### Community 52 - "Community 52"
Cohesion: 0.17
Nodes (11): Archived Follow-Up Ideas, DQN vs Random Forest, Historical Interpretation, Historical Summary Table, Legacy Architecture Notes, Main Takeaways, NSL-KDD Historical Experiments, Reward-System Sensitivity (+3 more)

### Community 53 - "Community 53"
Cohesion: 0.17
Nodes (11): Administrative: Citation Key Mismatches in Existing Draft, Citation Notes, Cumulative Citation Status, Second revised pass — 2026-05-16, Section A — CICIDS2017 Quality Concerns, Section B — Canonical Feature Schema and the Case for a Fixed Representation, Section C — Reinforcement Learning for Intrusion Detection: Named Prior Works, Section D — The Dataset-as-Environment Design: Named Precedent (+3 more)

### Community 54 - "Community 54"
Cohesion: 0.18
Nodes (10): Agent Workflows and Context Hierarchy, Building and Running, code:bash (pip install -r requirements.txt), code:bash (# Smoke test (quick run)), code:bash (# Run validation checks A, B, and C), code:bash (python scripts/predict_real_traffic_v2.py \), Development Conventions, Knowledge Graph (`graphify`) (+2 more)

### Community 55 - "Community 55"
Cohesion: 0.18
Nodes (10): code:text (runs/phase2/<RUN_ID>/), Current Open Risk, Data and Repo Safety, Lab Safety, Maintained Entry Point, Operational Guardrails, Out of Scope, Phase 2 Context (+2 more)

### Community 56 - "Community 56"
Cohesion: 0.18
Nodes (10): 01 — Fundamentos y objetivo del TFG, 1) Qué problema resuelve el proyecto, 2) Objetivo real del trabajo, 3) Estructura por fases, 4) Qué está implementado y qué no, 5) Mensaje fuerte para defensa, Fase 1 (más madura), Fase 2 (implementada pero abierta en robustez) (+2 more)

### Community 57 - "Community 57"
Cohesion: 0.18
Nodes (10): code:text (Escribe la subsección "Justificación y posicionamiento del t), Fuentes que sostienen el hueco, Gap y posicionamiento defendible para un TFG sobre defensa cibernética con RL, Handoff para Codex, Párrafos sugeridos para la tesis, Qué está débilmente validado en la literatura, Qué está parcialmente estudiado pero sigue siendo limitado, Qué está ya bien estudiado (+2 more)

### Community 58 - "Community 58"
Cohesion: 0.18
Nodes (10): code:text (Write the subsection "Justification and positioning of the w), Defensible Gap and Positioning for a Cyber Defense TFG with RL, Handoff for Codex, Sources supporting the gap, Suggested paragraphs for the thesis, What gap can your TFG reasonably claim, What is already well-studied, What is partially studied but remains limited (+2 more)

### Community 59 - "Community 59"
Cohesion: 0.18
Nodes (10): 1. Citation Key Convention, 2. Must-Cite Source Table, 3. Citation Placement Plan, 4. BibTeX TODOs, Citation Plan, code:text (AuthorYearShortTopic), Duplicated citation keys, Sources needing author/year verification (+2 more)

### Community 60 - "Community 60"
Cohesion: 0.18
Nodes (10): CICIDS2017, positioning, and integration aids, code:bibtex (@inproceedings{Cardenas2006IDSFramework,), code:bibtex (@inproceedings{Liu2022ErrorPrevalenceNIDS,), code:bibtex (@inproceedings{Henderson2018DRLMatters,), Evaluation pitfalls in ML, DL, and RL NIDS, Executive synthesis, Methodological Evaluation for a Flow-Level RL-Based NIDS Thesis, Metrics and operational meaning (+2 more)

### Community 61 - "Community 61"
Cohesion: 0.18
Nodes (11): CICFlowMeter-style flow features, Comparison of traffic-representation levels, Flow-level representation, Levels of representation, NetFlow/IPFIX-style records, Network Traffic Representation, Packet-level representation, Payload-level representation (+3 more)

### Community 62 - "Community 62"
Cohesion: 0.2
Nodes (9): Carga de CICIDS2017 y preprocesado, Deep research para defender TFG_CYBER_AI, Entorno RL y qué problema está resolviendo realmente, Entrenamiento, artefactos y reproducibilidad, Esquema canónico y contrato de datos, Fase 2, inferencia robusta y domain shift, Preguntas probables del tribunal, Qué he decidido investigar (+1 more)

### Community 63 - "Community 63"
Cohesion: 0.2
Nodes (9): Allowed Claims, Citation Discipline, Codex Handoff for Thesis Research Drafting, Forbidden Claims, Pre-Draft Checklist, Project Facts to Preserve, Required Reading, Suggested Drafting Structure for a Future Agent (+1 more)

### Community 64 - "Community 64"
Cohesion: 0.22
Nodes (8): AGENTS.md — TFG_CYBER_AI instructions for Codex, Anti-Leakage Rules, Documentation Rules, graphify, Project Invariants, Read This Before Making Changes, Reproducibility Expectations, Training and Validation Rules

### Community 65 - "Community 65"
Cohesion: 0.22
Nodes (8): 04 — Validación y lectura correcta de resultados, 1) Por qué no basta con una accuracy alta, 2) Check A (evaluación directa), 3) Check B (anti-leakage con etiquetas barajadas), 4) Check C (split duro por CSV/día), 5) Leave-one-exact-CSV-out, 6) Cómo explicar contradicciones aparentes en métricas, 7) Regla metodológica para el tribunal

### Community 66 - "Community 66"
Cohesion: 0.22
Nodes (8): Advisor / Tutor Questions, Claims Needing Verification, Experiments Needed, Explicit TODO List, Missing Sources, Research Gaps and TODOs, Uncertain Citations, Writing Decisions Needed

### Community 67 - "Community 67"
Cohesion: 0.46
Nodes (7): _find_event_dirs(), main(), parse_args(), _plot_scalar(), _read_scalars(), _safe_filename(), _update_artifact_manifest()

### Community 68 - "Community 68"
Cohesion: 0.25
Nodes (8): Bot-IoT, CICIDS2017 and CSE-CIC-IDS2018, Comparison of key NIDS datasets, Legacy datasets: KDDCup99 and NSL-KDD, Newer datasets and surveys, Public NIDS Datasets, ToN_IoT, UNSW-NB15

### Community 69 - "Community 69"
Cohesion: 0.29
Nodes (8): Binary PERMIT/BLOCK Classification, Cost-Sensitive IDS Reward, Dataset-as-Environment, QRDQN Policy Agent, Research2.md, report-classification-dossier.md, report-qrdqn-deep-distributional-rl.md, report-reward-and-cost-sensitive-design-dossier.md

### Community 70 - "Community 70"
Cohesion: 0.29
Nodes (7): CICIDS2017: Purpose, Structure, and Critiques, Common misuse patterns, Corrected or reconstructed versions, Known problems and critiques, Original purpose and structure, PCAPs vs generated CSVs and the CICFlowMeter pipeline, Responsible use as an internal benchmark

### Community 71 - "Community 71"
Cohesion: 0.29
Nodes (7): Cost_Sensitive_Reward, Phase_2_Inference, QRDQN, RLDatasetDefenderEnv, deep-research-report1, deep-research-report2, state_of_the_art_todo

### Community 72 - "Community 72"
Cohesion: 0.29
Nodes (7): CICIDS2017 Dataset, External Validation (Lab Traffic), Methodological Risks in NIDS, Research3.md, Temporal/Scenario Splits, report-deep-dive.md, report-source-map.md

### Community 73 - "Community 73"
Cohesion: 0.33
Nodes (5): Document Roles, Documentation Index, Language Policy, Notes, Reading Order

### Community 74 - "Community 74"
Cohesion: 0.33
Nodes (5): Investigación profunda para defensa (paquete multiarchivo), Mapa rápido de entradas técnicas del repositorio, Objetivo de este paquete, Orden recomendado de lectura, Relación con tus investigaciones previas

### Community 75 - "Community 75"
Cohesion: 0.33
Nodes (5): File Inventory, Purpose, Recommended Reading Order, Research Folder Index, Safety Notes

### Community 76 - "Community 76"
Cohesion: 0.33
Nodes (5): Drafting Guardrails, Recommended Chapter Structure, Safe Thesis Positioning, Section Plans, State of the Art Handoff

### Community 77 - "Community 77"
Cohesion: 0.33
Nodes (6): CICFlowMeter vs NetFlow-like schemas, Diversity of feature sets, Feature Engineering and Feature Standardisation in NIDS, Feature leakage risks and identifiers, Feature selection in NIDS, Missing, infinite, and categorical values

### Community 78 - "Community 78"
Cohesion: 0.33
Nodes (6): Alerting vs active response, Anomaly-based detection, IDS and NIDS Fundamentals, IDS vs IPS and deployment modes, Signature-based detection, Specification-based and hybrid detection

### Community 79 - "Community 79"
Cohesion: 0.33
Nodes (6): CICIDS2017-specific discussion, Feature engineering and standardisation, IDS/NIDS fundamentals, Network traffic representation, Public NIDS datasets, What to Add to the Current State of the Art

### Community 80 - "Community 80"
Cohesion: 0.4
Nodes (4): Contents, Experiment Archive, Notes, Status Labels Used Here

### Community 81 - "Community 81"
Cohesion: 0.5
Nodes (3): Overview, References, State of the Art on NIDS and Network-Flow Representations for a Flow-Level RL Defender

## Ambiguous Edges - Review These
- `Cost-Sensitive IDS Reward` → `Dataset-as-Environment`  [AMBIGUOUS]
   · relation: unknown

## Knowledge Gaps
- **941 isolated node(s):** `analyze_duplicates.py — Phase-0 / Task A1 (read-only analysis).  Quantifies, WIT`, `View a 2-D float32 array as a 1-D array of byte-exact per-row records.`, `predict_real_traffic_v2.py — Robust Phase 2 offline inference pipeline.  Loads`, `If time columns look like seconds (median < 1), convert to microseconds.`, `Load QRDQN model, falling back to DQN if sb3_contrib is unavailable.` (+936 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **12 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `Cost-Sensitive IDS Reward` and `Dataset-as-Environment`?**
  _Edge tagged AMBIGUOUS (relation: related to) - confidence is low._
- **Why does `load_cicids2017_split()` connect `Community 0` to `Community 37`, `Community 14`?**
  _High betweenness centrality (0.011) - this node is a cross-community bridge._
- **Why does `map_to_canonical()` connect `Community 20` to `Community 0`, `Community 4`?**
  _High betweenness centrality (0.008) - this node is a cross-community bridge._
- **Why does `_prepare_cicids_features()` connect `Community 0` to `Community 20`?**
  _High betweenness centrality (0.007) - this node is a cross-community bridge._
- **Are the 6 inferred relationships involving `RLDatasetDefenderEnv` (e.g. with `ProgressCallback` and `_evaluate_f1()`) actually correct?**
  _`RLDatasetDefenderEnv` has 6 INFERRED edges - model-reasoned connections that need verification._
- **What connects `analyze_duplicates.py — Phase-0 / Task A1 (read-only analysis).  Quantifies, WIT`, `View a 2-D float32 array as a 1-D array of byte-exact per-row records.`, `predict_real_traffic_v2.py — Robust Phase 2 offline inference pipeline.  Loads` to the rest of the system?**
  _941 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.06 - nodes in this community are weakly interconnected._