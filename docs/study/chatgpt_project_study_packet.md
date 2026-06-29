# TFG_CYBER_AI ChatGPT Study Packet
**Version 1.0.1** · Author: Rivero + Codex · Date: 2026-06-29 · Scope: single-file study context for ChatGPT

---

## AI READING INSTRUCTION
**[SPEC]**
- This document is a self-contained study packet for the project `TFG_CYBER_AI`.
- Read all `[SPEC]` blocks as authoritative project facts.
- Read `[NOTE]` blocks to teach, explain, quiz, and translate technical ideas into plain Spanish.
- Treat `[?]` blocks as limitations, open risks, or lower-confidence interpretation.
- Do not treat this project as a production-ready network blocker.
- Do not claim full sequential RL: the current training setup uses `gamma=0.0`, so the implemented formulation is best defended as a cost-sensitive contextual bandit built with RL machinery.
- Do not claim QRDQN generally beats Random Forest: Random Forest wins on the easy random split; QRDQN holds up better on the committed day-split artifact.
- Do not say Phase 2 is solved: Phase 2 remains sensitive to `domain shift` and exact run artifacts.
- If GitHub app/repo access is available, use it to inspect the current repository files and committed artifacts when checking a claim.
- If this packet and the live repository disagree, prefer current repo code plus committed run artifacts, then explain the discrepancy.

**[NOTE]**
Use this file to help the author study. Your role is not to flatter or merely summarize. Your role is to teach, interrogate, correct, and force precise understanding.

---

## 1. AI Reading Instruction And ChatGPT Tutor Prompt
**[NOTE] Ready-to-paste ChatGPT tutor prompt**
```text
Act as a rigorous technical tutor for my TFG project. Use the study packet below as your starting source of truth. You also have GitHub app access to the repository, so use it when useful to verify current code, docs, and committed run artifacts.

Your goals:
1. Teach me the project progressively, from simple explanations to technical detail.
2. Ask Socratic questions and wait for my answer.
3. Grade my answers from 0 to 5.
4. Identify vague wording, conceptual errors, and unsupported claims.
5. Make me explain each idea twice: first with technical language, then "en cristiano".
6. If I make a mistake or I do not know, teach the concept and then ask a follow-up question.
7. Do not accept vague answers about:
   - gamma=0.0 and why the formulation is a contextual bandit.
   - the 76+76 observation vector and missingness mask.
   - why Random Forest is a serious baseline and wins on the random split.
   - why Phase 2 is not production validation.
   - what data leakage is and how the project tries to prevent it.
8. Keep me honest: distinguish artifact-backed facts, current code defaults, historical runs, and open limitations.
9. If the pasted packet and the live GitHub repository disagree, prefer current repo code plus committed artifacts, and tell me exactly what changed.

Start by asking me to explain the project in two minutes without reading notes.
```

---

## 2. Project Overview In 1 Minute, 5 Minutes, And 20 Minutes
**[SPEC]**
- Project goal: build and evaluate an RL-based cybersecurity defender that observes flow-based network features and decides:
  - `0 = PERMIT`
  - `1 = BLOCK`
- The project has two major phases:
  - Phase 1: offline training and validation on datasets.
  - Phase 2: offline inference on flow features extracted from traffic captured in a private lab.
- Current maintained model path:
  - Dataset: CICIDS2017.
  - Observation: 152 dimensions.
  - Agent: QRDQN.
  - Environment: custom Gymnasium environment `RLDatasetDefenderEnv`.
  - Main inference script for Phase 2: `scripts/predict_real_traffic_v2.py`.
- Current non-goals:
  - No active real-time blocking with `iptables` or `nftables`.
  - No multi-agent or adversarial RL.
  - No production deployment claim.
  - No fully completed Phase 2 calibration guarantee for real benign traffic.

**[NOTE] 1-minute explanation**
Mi TFG construye un defensor de ciberseguridad offline que mira estadisticas de flujos de red y decide si permitir o bloquear cada flujo. Uso CICIDS2017 como dataset principal, convierto cada flujo a un vector fijo de 152 valores, y entreno un agente QRDQN dentro de un entorno Gymnasium. La decision es binaria: `PERMIT` o `BLOCK`. La recompensa penaliza mucho mas dejar pasar un ataque que bloquear trafico benigno, porque en seguridad un falso negativo suele ser mas grave que una falsa alarma. El resultado principal funciona muy bien en el split aleatorio de CICIDS2017, pero soy honesto: con `gamma=0.0` esto no es RL secuencial completo, sino un `bandit contextual` sensible al coste. Ademas, Phase 2 sobre trafico de laboratorio sigue teniendo `domain shift`, asi que no puedo venderlo como defensa real en produccion.

**[NOTE] 5-minute explanation**
El problema parte de trafico de red convertido en flujos. Un flujo no es el contenido de los paquetes, sino una ficha estadistica de una conversacion: duracion, bytes, paquetes, tiempos entre llegadas, flags TCP, tasas, longitudes, etc. El proyecto usa un esquema canonico de 76 caracteristicas numericas de flujo. Para que el modelo sepa si una caracteristica estaba realmente presente o fue imputada, se concatena una mascara de ausencia de otras 76 posiciones. Por eso la observacion final tiene 152 dimensiones: `[76 valores | 76 mascara]`.

Sobre CICIDS2017, el mapeo cubre las 76 caracteristicas, asi que la mascara es constante a 1 y no aporta informacion al resultado interno. Su sentido aparece cuando se pasa a otros dominios, como Phase 2, donde pueden faltar columnas y la mascara marca que ciertos ceros son imputados.

El entorno `RLDatasetDefenderEnv` recorre muestras etiquetadas y presenta una observacion al agente. El agente elige `PERMIT` o `BLOCK`. La recompensa codifica una matriz de costes: ataque bloqueado `+1.5`, benigno bloqueado `-2.0`, ataque permitido `-5.0`, benigno permitido `0.0`. Esto empuja a reducir ataques no detectados sin ocultar el coste de falsas alarmas.

El algoritmo es QRDQN, una version distribucional de DQN. En vez de aprender solo el valor medio de cada accion, aprende una distribucion de retornos mediante cuantiles. En este proyecto se usan 200 cuantiles. Aun asi, como `gamma=0.0`, no hay planificacion temporal: el objetivo de aprendizaje se reduce a la recompensa inmediata. Por tanto, la defensa honesta es que uso la maquinaria de RL para un problema `cost-sensitive` de un paso, no que tengo un agente secuencial autonomo.

Los resultados principales estan respaldados por artefactos. El run oficial `MAIN_qrdqn_cicids2017_canonical_full_random_20260609_193655` usa todo CICIDS2017, 3,000,000 timesteps, train/test de 2,264,594 / 566,149 filas, y logra accuracy `0.99381`, recall de ataque `0.99536` y F1 de ataque `0.98445`. Pero Random Forest, bajo el mismo protocolo en split aleatorio, obtiene F1 de ataque `0.99676`, asi que no puedo decir que QRDQN sea superior en general. En cambio, en el split duro por dia, Random Forest colapsa mucho mas que QRDQN: RF recall de ataque `0.08135`, frente a QRDQN Check C recall `0.52954`.

Phase 2 hace inferencia offline sobre trafico capturado en laboratorio. El artefacto principal etiquetado `P2v2_pred_20260610_161231_MAIN` obtiene accuracy `0.991862` y recall de ataque `0.988452`, pero es un benchmark de laboratorio controlado, con validez externa limitada. Otros artefactos benign-only oscilan entre bloquear todo y permitir todo. Eso demuestra sensibilidad al dominio y al preprocesado. La conclusion correcta es: buen pipeline offline y buen ejercicio metodologico, pero no produccion.

**[NOTE] 20-minute explanation skeleton**
- Min 0-2: problema y contribucion.
- Min 2-5: que es un flujo, por que se usan flujos, que es CICIDS2017.
- Min 5-8: esquema canonico de 76 features, mascara de 76, observacion 152.
- Min 8-10: anti-leakage, etiquetas binarias, limpieza, split y scaler.
- Min 10-13: entorno RL: estado, accion, recompensa, matriz de coste.
- Min 13-16: QRDQN: DQN, cuantiles, replay buffer, target network, epsilon-greedy.
- Min 16-18: punto clave: `gamma=0.0`, contextual bandit, no RL secuencial pleno.
- Min 18-21: entrenamiento MAIN, hiperparametros, reproducibilidad, RUN_ID.
- Min 21-25: resultados: MAIN, bootstrap CI, Checks A/B/C, Random Forest.
- Min 25-28: Phase 2, domain shift, inferencia offline, limitaciones.
- Min 28-30: que no puedo afirmar y como defenderia objeciones del tutor.

---

## 3. Architecture Map: Datasets, Canonical Schema, Preprocessing, Environment, QRDQN, Validation, Phase 2 Inference
**[SPEC]**
| Subsystem | Main file / artifact | Responsibility |
|---|---|---|
| Canonical schema | `src/canonical_schema.py` | Defines `FEATURES_CANON`, canonical mapping, imputation, mask, 152-dim observation names. |
| CICIDS2017 adapter | `src/load_cicids2017.py` | Loads official CSVs, cleans rows, drops leakage-prone columns, maps to canonical schema, splits, scales. |
| Historical NSL-KDD adapter | `src/load_nsl_kdd.py` | Retained for reference and historical benchmarking; not final Phase 2 path. |
| RL environment | `src/rl_defender_env.py` | Defines Gymnasium environment with `PERMIT/BLOCK` actions and asymmetric rewards. |
| Training | `src/train_rl_defender.py` | Trains QRDQN on canonical CICIDS2017 observations and writes run artifacts. |
| Validation A/B/C | `src/validate_checks.py` | Direct evaluation, shuffled-label anti-leakage check, hard CSV/day split. |
| Leave-one-exact-CSV-out | `src/validate_leave_one_csv_out.py` | Implements exact CSV holdout validation; no committed full artifact yet. |
| Random Forest baseline | `src/baseline_random_forest.py` | Supervised baseline under same canonical/scaled protocol. |
| Phase 2 inference | `scripts/predict_real_traffic_v2.py` | Offline inference over extracted lab-flow CSVs with robust loading and diagnostics. |
| Results snapshot | `docs/results.md` | Maintained artifact-backed result summary. |

**[NOTE]**
Piensa en el proyecto como una cadena:

```text
CSV de flujos -> limpieza -> esquema canonico -> split -> scaler -> entorno RL -> QRDQN -> metricas -> inferencia offline Phase 2
```

El valor del proyecto no esta en inventar QRDQN ni en inventar CICIDS2017. Esta en conectar estas piezas de forma reproducible, con una formulacion `PERMIT/BLOCK`, una recompensa sensible al coste, un protocolo de validacion y una discusion honesta de las limitaciones.

**[SPEC]**
- The maintained documentation rule is: if documentation and code disagree, prefer current code plus run artifacts, then update documentation.
- `graphify-out/needs_update` exists in this workspace, so graph output should not be treated as authoritative for this packet.

---

## 4. Core Invariants: 76 Canonical Features, 152 Observations, Mask Semantics, Labels, Adapter Contract, Anti-Leakage Policy
**[SPEC]**
- `FEATURES_CANON` contains exactly 76 canonical flow features.
- `NUM_OBSERVATION_FEATURES = NUM_CANONICAL_FEATURES * 2`.
- Final observation size is 152:
  - first 76 positions: canonical feature values.
  - next 76 positions: missingness mask values.
- Observation layout:
  - `obs = [x_1..x_76, m_1..m_76]`
- Mask semantics:
  - `1 = present / valid`
  - `0 = missing / imputed`
- Default imputation value:
  - `DEFAULT_IMPUTATION_VALUE = 0.0`
- Expected data types:
  - `X = float32`
  - `y = int64`
- Label convention:
  - `0 = BENIGN`
  - `1 = ATTACK`
- Dataset adapter contract:
  - `(X_train, y_train, X_test, y_test, scaler, feature_names)`

**[SPEC]**
Anti-leakage policy: the following must not be used as model features:
- source or destination IPs.
- absolute timestamps.
- Flow IDs or unique identifiers.
- direct port fields when they act as label proxies.

**[NOTE]**
En cristiano:
- Las 76 features son las medidas del flujo.
- Las otras 76 posiciones son una libreta de fiabilidad: dicen si cada medida es real o fue rellenada.
- La forma 152 nunca cambia, aunque falten columnas.
- Si falta una variable, no se borra del vector: se rellena con `0.0` y su mascara vale `0`.
- Esto evita que el modelo confunda "el valor real era cero" con "no tenia ese dato y lo he puesto a cero".

**[NOTE]**
Matiz importante para defender: en CICIDS2017 nativo, el mapeo cubre las 76 features. Eso significa que la mascara queda a 1 para todas las posiciones mapeadas y no explica el alto rendimiento interno. No digas que la mascara "mejora" el resultado de CICIDS2017. Di que mantiene una interfaz estable y se vuelve informativa en cambio de dominio, cuando faltan columnas.

**[?]**
La utilidad real de la mascara en Phase 2 depende de cuantas columnas falten, de como se extraigan los flujos y de si el modelo ha visto patrones comparables durante entrenamiento. Es una defensa metodologica razonable, no una garantia de robustez.

---

## 5. Full Data Pipeline: CICIDS2017 CSVs To Cleaning To Canonical Mapping To Split To Scaler To Observation Vector
**[SPEC]**
- Primary dataset: CICIDS2017.
- Default loader policy uses the 8 official CICIDS2017 CSV files.
- By default, `allow_non_official_csvs=False`.
- Missing official CSVs raise `FileNotFoundError`.
- CSV loading uses:
  - `chunksize = 250_000`
  - `low_memory=True`
  - `encoding_errors="ignore"`
- Label column:
  - `Label`
- Label binarization:
  - label normalized to uppercase/stripped.
  - `BENIGN -> 0`.
  - anything else -> `1`.
- Default random split:
  - stratified train/test split.
  - `test_size = 0.2`.
  - `random_state = 42`.
- Scaling:
  - `StandardScaler`.
  - fitted only on train.
  - applied to test with train-fitted statistics.

**[SPEC]**
- Current loader supports:
  - random stratified split.
  - CSV/day pattern split.
  - exact-file split for leave-one-exact-CSV-out validation.
  - train-only subsampling with `train_max_rows`.
- `train_max_rows` subsamples only train after the split.
- `train_max_rows` is deterministic, stratified, and nested through `stratified_nested_prefix_v1`.
- Test partition stays byte-identical for comparable `train_max_rows` benchmark runs under the same seed and full preset.

**[NOTE] Data pipeline in plain language**
1. Cargo los CSV oficiales.
2. Quito o ignoro columnas peligrosas para fuga, como IPs, puertos, `Flow ID` y `Timestamp`.
3. Convierto las etiquetas a binario: benigno o ataque.
4. Limpio valores numericos problematicos: `Inf`, `NaN`, columnas no numericas.
5. Mapeo nombres originales de CICIDS2017 a nombres canonicos.
6. Relleno ausencias con `0.0`.
7. Creo la mascara de presencia.
8. Concateno features y mascara para obtener 152 dimensiones.
9. Divido en train/test.
10. Ajusto `StandardScaler` solo con train.
11. Entreno y evaluo.

**[NOTE] Why fit the scaler only on train**
Si ajusto el scaler con train + test, el test influye en las medias y desviaciones que vera el modelo. Eso contamina la evaluacion: el test deja de ser totalmente independiente. En cristiano: seria como preparar el examen despues de haber visto una pista estadistica de las preguntas del examen.

**[NOTE] Why ports are removed**
Los puertos pueden parecer informacion legitima de red, pero en CICIDS2017 pueden actuar como atajo hacia la etiqueta. Por ejemplo, ciertos ataques tienden a ocurrir en puertos concretos. Si dejo el puerto, el modelo podria aprender "puerto X = ataque" en vez de aprender comportamiento generalizable. Quitar puertos reduce rendimiento aparente en algunos casos, pero aumenta seguridad metodologica.

**[NOTE] What a network flow is**
Un flujo de red es el resumen estadistico de una comunicacion entre maquinas. No miro cada paquete como texto o contenido, sino la ficha resumen: duracion, paquetes, bytes, flags, tiempos, tasas. En cristiano: no leo la conversacion; miro sus metadatos y su comportamiento.

---

## 6. RL Formulation: State, Action, Reward, Environment, Offline Setting, `gamma=0.0`, Contextual Bandit Limitation
**[SPEC]**
- Environment class: `RLDatasetDefenderEnv`.
- Environment type: Gymnasium `gym.Env`.
- Observation space:
  - `spaces.Box(low=-inf, high=+inf, shape=(n_features,), dtype=float32)`.
- Action space:
  - `spaces.Discrete(2)`.
- Actions:
  - `0 = PERMIT`
  - `1 = BLOCK`
- Reward defaults:
  - true positive: attack blocked -> `tp = +1.5`
  - false positive: benign blocked -> `fp = -2.0`
  - false negative: attack permitted -> `fn = -5.0`
  - benign permitted: `omission = 0.0`
- Current code defaults use the same reward config in training and validation.

**[SPEC]**
- `gamma=0.0` in the main QRDQN profiles.
- With `gamma=0.0`, the bootstrap future-return term is removed.
- The target is effectively immediate reward only.
- The current formulation is best described as a cost-sensitive contextual bandit implemented with RL infrastructure.

**[NOTE] State, action, reward**
- Estado: el vector de 152 dimensiones de un flujo.
- Accion: permitir o bloquear.
- Recompensa: premio/castigo segun etiqueta real y accion.
- Politica: regla aprendida por el modelo para escoger `PERMIT` o `BLOCK`.

**[NOTE] Why this is still framed as RL**
No se entrena con entropia cruzada como un clasificador supervisado tradicional. El agente aprende valores de accion a partir de una recompensa. La diferencia defendible es que el coste de seguridad vive explicitamente en la recompensa y la decision es el `argmax` del valor aprendido. Pero hay que reconocer el limite: no hay una dinamica temporal real.

**[NOTE] Why this is not full sequential RL**
En un MDP secuencial real, mi accion cambia el estado futuro. Si bloqueo un flujo, eso podria cambiar el comportamiento del atacante, reducir trafico posterior, activar otra defensa, etc. En este proyecto, el dataset es estatico: el siguiente flujo no depende de mi accion anterior. Por eso `gamma=0.0` es coherente: no hay futuro causal que valorar.

**[NOTE] Best answer if challenged: "Is this classification disguised as RL?"**
Respuesta fuerte:

> En parte la objecion es justa. Con `gamma=0.0`, mi formulacion no es RL secuencial pleno, sino un contextual bandit cost-sensitive. Pero no es simplemente un clasificador estandar con entropia cruzada: uso un entorno Gymnasium, acciones `PERMIT/BLOCK`, recompensas asimetricas, aprendizaje de valor y una cabeza distribucional QRDQN. La aportacion no es fingir que hay planificacion temporal, sino formular la decision defensiva como aprendizaje de valor sensible al coste y dejar una arquitectura que podria extenderse a mas acciones o dinamica secuencial real.

**[?]**
La formulacion actual no prueba que QRDQN sea necesario. Prueba que el pipeline RL es viable y evaluable. La necesidad metodologica de RL seria mas fuerte con acciones defensivas multiples, consecuencias temporales reales, o un entorno que responda a las acciones del defensor.

---

## 7. QRDQN Deep Dive: DQN Vs QRDQN, Quantiles, Replay Buffer, Target Network, Epsilon-Greedy, Quantile-Huber Loss
**[SPEC]**
- Algorithm: QRDQN from `sb3_contrib`.
- QRDQN means Quantile Regression DQN.
- DQN learns an expected Q-value per action.
- QRDQN learns a distribution of returns represented by quantiles.
- Main experiment profile uses:
  - `n_quantiles = 200`
  - `learning_rate = 5e-5`
  - `buffer_size = 1_000_000`
  - `learning_starts = 50_000`
  - `batch_size = 2048`
  - `gamma = 0.0`
  - `gradient_steps = 20`
  - `target_update_interval = 10_000`
  - `exploration_initial_eps = 1.0`
  - `exploration_final_eps = 0.02`
  - `exploration_fraction = 0.10`
- QRDQN selects actions by collapsing quantiles to mean value per action and taking the best action.

**[NOTE] DQN in plain Spanish**
DQN aprende una puntuacion para cada accion. Si el estado es un flujo y las acciones son permitir o bloquear, DQN aprende algo como: "en este flujo, permitir vale X y bloquear vale Y". Escoge la accion con mayor valor esperado.

**[NOTE] QRDQN in plain Spanish**
QRDQN no aprende solo una media. Aprende una distribucion de posibles resultados para cada accion, representada por cuantiles. En cristiano: en vez de decir "de media bloquear vale 1.2", intenta aprender el reparto de resultados posibles de bloquear. Eso puede ser interesante en seguridad porque el riesgo no siempre se resume bien con una media.

**[NOTE] Quantiles**
Un cuantil es un punto que divide una distribucion. Con 200 cuantiles, el modelo representa el retorno con 200 marcas. Si el retorno de una accion puede variar, esos cuantiles dibujan su forma aproximada.

**[NOTE] Replay buffer**
QRDQN es `off-policy`. Guarda experiencias pasadas en un `replay buffer`: observacion, accion, recompensa, siguiente observacion y fin de episodio. Luego entrena muestreando minibatches aleatorios. Esto evita aprender solo de ejemplos consecutivos muy parecidos.

**[NOTE] Target network**
La `target network` es una copia de la red usada para estabilizar los objetivos de aprendizaje. En este proyecto sigue existiendo, pero con `gamma=0.0` su papel se reduce mucho porque el target no necesita valor futuro.

**[NOTE] Epsilon-greedy**
`epsilon-greedy` controla exploracion vs explotacion. Al principio, con epsilon alto, el agente prueba acciones aleatorias. Luego epsilon baja y el agente usa mas lo que cree mejor. En el main profile, epsilon empieza en `1.0` y termina en `0.02`.

**[NOTE] Quantile-Huber loss**
La perdida `quantile-Huber` combina:
- Huber loss: penaliza errores grandes de forma mas robusta que error cuadratico puro.
- ponderacion asimetrica de cuantiles: necesaria para aprender puntos concretos de una distribucion, no solo la media.

**[NOTE] Critical nuance with gamma=0.0**
Aunque QRDQN tiene replay buffer, red objetivo y distribucion de retornos, el `gamma=0.0` elimina el componente de futuro. Por eso no debes vender "asignacion temporal de credito" ni "planificacion a largo plazo". La parte defendible es el aprendizaje de valor inmediato bajo coste asimetrico.

---

## 8. Training And Reproducibility: RunPod/GPU Setup, Seeds, Hyperparameters, Artifacts, `RUN_ID`
**[SPEC]**
- Real main training ran on GPU.
- Main run:
  - `RUN_ID = MAIN_qrdqn_cicids2017_canonical_full_random_20260609_193655`
  - algorithm: QRDQN.
  - dataset: CICIDS2017.
  - observation size: 152.
  - train/test: 2,264,594 / 566,149.
  - total timesteps: 3,000,000.
  - device: `cuda`.
  - reward config: `tp=1.5, fp=-2.0, fn=-5.0, omission=0.0`.
- Main run metrics and config are artifact-backed under `runs/cicids2017/MAIN_qrdqn_cicids2017_canonical_full_random_20260609_193655/`.

**[SPEC]**
- Default seed: `42`.
- Training seeds:
  - Python `random`.
  - NumPy.
  - Torch.
  - CUDA, when available.
  - vectorized environment seed.
  - QRDQN constructor seed.
- The global seed does not redefine the fixed dataset split; the split uses its own `random_state`.

**[SPEC]**
- Reproducibility expectations:
  - meaningful runs should persist `config.json`.
  - metrics should be stored as `metrics.json` or `validation_results.json`.
  - exact `RUN_ID` must be kept.
  - preprocessing, clipping, scaling, reward values, and split logic changes should be documented.

**[NOTE] Why `RUN_ID` matters**
Un `RUN_ID` es la identidad exacta de un experimento. Sin `RUN_ID`, una cifra como "recall 0.995" no se puede auditar. Con `RUN_ID`, puedo ir al artefacto, ver config, metricas, modelo, scaler, fecha y protocolo.

**[NOTE] Environment story**
El entrenamiento pesado se deja a RunPod/GPU. Localmente se deben preferir comprobaciones estaticas, tests ligeros o comandos reproducibles. No hay que inventar resultados de entrenamiento si no se ejecutaron.

**[NOTE] Historical run caution**
Los runs C0x son historicos o pre-design probes. C03 puede tener metricas mas bonitas que MAIN, pero no es el resultado oficial porque uso `max_rows=500000` y un test de 100,000 filas con mezcla distorsionada. No lo presentes como "mejor resultado oficial".

---

## 9. Results And Validation: MAIN Run, Check A/B/C, Bootstrap CIs, Random Forest Baseline, Phase 2 Artifacts
**[SPEC] MAIN official run**
| Field | Value |
|---|---|
| RUN_ID | `MAIN_qrdqn_cicids2017_canonical_full_random_20260609_193655` |
| Algorithm | QRDQN |
| Dataset | CICIDS2017 |
| Observation size | 152 |
| Train / test | 2,264,594 / 566,149 |
| Timesteps | 3,000,000 |
| Accuracy | `0.99381` |
| Precision attack | `0.97378` |
| Recall attack | `0.99536` |
| F1 attack | `0.98445` |
| Reward config | `tp=1.5, fp=-2.0, fn=-5.0, omission=0.0` |

**[SPEC] MAIN confusion matrix and CIs**
- Confusion matrix:
  - `tn = 451,631`
  - `fp = 2,989`
  - `fn = 518`
  - `tp = 111,011`
  - total = 566,149.
- 95% bootstrap confidence intervals over the fixed seed-42 test set:
  - recall attack: `[0.99495, 0.99575]`
  - FPR: `[0.00634, 0.00681]`
  - FNR: `[0.00425, 0.00505]`
  - balanced accuracy: `[0.99416, 0.99462]`
  - MCC: `[0.98004, 0.98130]`
  - F1 attack: `[0.98394, 0.98496]`
- These CIs quantify test-set resampling precision for this one trained model.
- They do not measure training-seed variability.

**[SPEC] Validation checks**
| Check | Artifact | Purpose | Key result |
|---|---|---|---|
| Check A | `runs/validation/VAL_checks_A_20260212_235443/validation_results.json` | Direct prediction vs `y_test` | Accuracy `0.9939`, recall attack `0.99979`, F1 attack `0.99365`. |
| Check B | `runs/validation/VAL_checks_B_20260212_235736/validation_results.json` | Shuffled-label anti-leakage | Shuffled accuracy `0.4773`, leakage detected `false`. |
| Check C | `runs/validation/VAL_checks_C_20260213_004847/validation_results.json` | Hard CSV/day split | Accuracy `0.84135`, recall attack `0.52954`, F1 attack `0.62578`. |

**[SPEC] Random Forest baseline**
- Baseline run: `rf_cicids2017_canonical_20260628_024735`.
- Same-protocol comparison:
  - canonical observation.
  - scaled features.
  - `class_weight="balanced"`.
- Random split:
  - test rows: 566,149.
  - accuracy: `0.99872`.
  - F1 attack: `0.99676`.
  - precision attack: `0.99501`.
  - recall attack: `0.99853`.
  - RF marginally beats QRDQN on the random split.
- Day split:
  - test rows: 1,162,213.
  - accuracy: `0.76913`.
  - F1 attack: `0.15005`.
  - precision attack: `0.96473`.
  - recall attack: `0.08135`.
  - RF attack recall collapses much more than QRDQN Check C on the same Mon/Tue/Wed -> Thu/Fri partition.
- Leave-one-out Wednesday:
  - RF-only committed artifact.
  - F1 attack: `0.01427`.
  - no committed QRDQN counterpart artifact.

**[SPEC] Phase 2 artifacts**
| Artifact | Traffic | Key result | Correct interpretation |
|---|---|---|---|
| `P2v2_pred_20260224_004121` | benign-only lab flow CSV | block rate `1.0`, allow rate `0.0` | Strong domain-shift / overblocking behavior. |
| `P2v2_pred_20260408_230318` | benign-only lab flow CSV | block rate `0.0`, allow rate `1.0` | Later run shows behavior changed; cite exact artifact. |
| `P2v2_pred_20260610_161231_MAIN` | labeled lab capture | accuracy `0.991862`, precision attack `0.97919`, recall attack `0.988452`, F1 attack `0.983801`, block rate `0.252364` | Operator-generated lab-capture benchmark; not production validation. |

**[NOTE] How to explain the results honestly**
El split aleatorio principal muestra que el pipeline aprende muy bien dentro de CICIDS2017. Pero el split por dia muestra que generalizar a dias o patrones distintos es mas dificil. Random Forest gana en el caso facil, asi que el valor del proyecto no puede ser "QRDQN siempre es mejor". El valor es: formulacion metodologica, evaluacion con costes, pipeline reproducible, validaciones anti-fuga y analisis honesto de generalizacion.

**[NOTE] What Check B proves and does not prove**
Check B con etiquetas barajadas ayuda a detectar fuga obvia: si el modelo siguiera rindiendo muy alto con etiquetas aleatorias, habria un atajo grave. Como no ocurre, apoya que no hay fuga evidente. Pero no prueba ausencia absoluta de todos los sesgos o duplicados del dataset.

**[NOTE] What Check C teaches**
Check C es duro porque separa por dia/CSV. Simula mejor la idea de aprender con unos dias y evaluar en otros. La caida de recall muestra que el split aleatorio probablemente sobreestima la generalizacion real.

---

## 10. Critical Limitations And Forbidden Claims: No Production Guarantee, No Autonomous Real-Time Blocking, No Sequential RL Claim, Phase 2 Domain Shift Unresolved
**[SPEC] Forbidden claims**
- Do not claim the system works in enterprise production networks.
- Do not claim active real-time blocking is implemented.
- Do not claim the current setup is full sequential RL.
- Do not claim QRDQN generally outperforms Random Forest.
- Do not claim Phase 2 is solved.
- Do not present C03 as the official best result.
- Do not mix CICIDS2017 internal metrics with Phase 2 lab metrics.
- Do not treat bootstrap CIs as training-seed stability.

**[SPEC] Current limitations**
- Dataset is static; actions do not affect future states.
- `gamma=0.0` removes future credit assignment.
- CICIDS2017 is old lab traffic and has known issues.
- Random row-wise split can overestimate performance in network datasets.
- Only one main training seed is documented for the main model.
- Phase 2 lab traffic has limited external validity.
- Phase 2 behavior changed across artifacts.
- No full committed leave-one-exact-CSV-out QRDQN artifact exists yet.

**[NOTE] Strong limitation answer**
La principal debilidad metodologica no es simplemente "faltan mas datos". Es que el entorno es estatico y la accion no cambia el futuro. Con `gamma=0.0`, el proyecto resuelve una decision independiente por flujo. Eso es defendible como contextual bandit cost-sensitive, pero no como defensa autonoma secuencial. Para acercarlo a una defensa real harian falta acciones con consecuencias, estado historico, adversario reactivo, latencia, throughput y enforcement real.

**[NOTE] Safe claims**
- "He construido un pipeline offline reproducible para estudiar decisiones `PERMIT/BLOCK` sobre flujos."
- "Uso QRDQN para aprender valores de accion bajo recompensa asimetrica."
- "Con `gamma=0.0`, la formulacion actual es un contextual bandit, no un MDP secuencial completo."
- "El resultado MAIN es fuerte en CICIDS2017 con split aleatorio, pero no basta para produccion."
- "Random Forest es un baseline fuerte y gana en random split; QRDQN resiste mejor en el Check C comprometido."
- "Phase 2 es inferencia offline en laboratorio con domain shift abierto."

**[NOTE] Unsafe claims**
- "Mi sistema bloquea ataques en tiempo real."
- "El modelo esta listo para una empresa."
- "QRDQN demuestra ser mejor que Random Forest."
- "La mascara mejora el resultado de CICIDS2017."
- "Los intervalos bootstrap prueban que cualquier reentrenamiento dara igual."
- "Check C demuestra produccion."

---

## 11. Tutor Question Bank: Expected Answer, What It Tests, Weak Answer Patterns, Corrected Answer
**[SPEC]**
- The original tutor-question file contains 50 comprehension questions plus 10 especially diagnostic questions.
- This section converts them into study prompts with expected reasoning.

**[NOTE] Scoring rubric for every answer**
- 5/5: precise, artifact-aware, technically correct, admits limitations.
- 4/5: mostly correct, one missing nuance.
- 3/5: understandable but generic; needs project-specific evidence.
- 2/5: repeats buzzwords without explaining mechanisms.
- 1/5: materially wrong or overclaims.
- 0/5: cannot answer or contradicts core invariants.

### A. General Comprehension
**[NOTE] Q1. Explain the project in two minutes without using title words.**
- Expected: defender observes network-flow statistics, decides allow/block, trained offline, cost-sensitive reward, QRDQN, CICIDS2017, Phase 2 offline lab inference, limitations.
- Tests: whether you understand the problem independently from the title.
- Weak answer: "It uses AI for cybersecurity."
- Corrected answer: "It maps flow statistics to a binary defensive decision and evaluates that decision under asymmetric security costs."

**[NOTE] Q2. Why not just a supervised classifier?**
- Expected: because the project wants explicit action values and asymmetric cost in reward; also compares against RF to avoid overclaiming.
- Tests: whether you can defend the RL framing without denying its bandit nature.
- Weak answer: "RL is more advanced."
- Corrected answer: "The defensible reason is cost-sensitive action-value learning, not algorithmic prestige."

**[NOTE] Q3. What are binary decisions here?**
- Expected: `0=PERMIT`, `1=BLOCK`; experimental labels/actions, not real firewall enforcement.
- Weak answer: "Attack or benign."
- Corrected answer: "The action is permit/block; the label is benign/attack."

**[NOTE] Q4. PERMIT/BLOCK experimental actions vs real blocking**
- Expected: in this project they are model decisions/predictions in offline evaluation; no production firewall consumes them.
- Weak answer: "It blocks attacks."
- Corrected answer: "It predicts what would be allowed or blocked; active enforcement is out of scope."

**[NOTE] Q5. Main contribution**
- Expected: methodological formulation and reproducible pipeline, with evaluation protocol; not new algorithm or dataset.
- Weak answer: "The contribution is QRDQN."
- Corrected answer: "QRDQN is reused; contribution is how the decision problem, canonical schema, rewards, and validation are assembled."

### B. Cybersecurity And Data
**[NOTE] Q6. What is a network flow?**
- Expected: statistical summary of communication between endpoints over a session; duration, packets, bytes, flags, rates, IAT.
- Weak answer: "A packet."
- Corrected answer: "A flow summarizes multiple packets; it is not the raw packet content."

**[NOTE] Q7. Why flows instead of deep packet inspection?**
- Expected: lighter, privacy-friendly, works with encrypted payloads, focuses on behavior.
- Weak answer: "Because it is easier."
- Corrected answer: "Ease is secondary; the technical reason is behavior-level metadata without payload inspection."

**[NOTE] Q8. Known CICIDS2017 problems**
- Expected: duplicates/repetition, lab/old traffic, class imbalance, possible extraction artifacts, leakage-prone fields, random split overestimation.
- Weak answer: "It is a standard dataset."
- Corrected answer: "It is standard but imperfect; high metrics need cautious interpretation."

**[NOTE] Q9. Why high CICIDS2017 result does not prove production**
- Expected: domain shift; enterprise traffic differs in services, users, attacks, extraction, timing and normal behavior.
- Weak answer: "Because more testing is needed."
- Corrected answer: "Because the distribution changes; this is not just sample size."

**[NOTE] Q10. Leakage variables**
- Expected: IPs, timestamps, Flow IDs, ports; also duplicates and data split leakage can create hidden shortcuts.
- Weak answer: "Only IPs."
- Corrected answer: "Leakage includes direct identifiers and indirect proxies like ports or split contamination."

### C. Preprocessing And Canonical Schema
**[NOTE] Q11. What does canonical schema of 76 variables mean?**
- Expected: fixed ordered list of 76 numeric flow features with standardized names independent of original dataset column names.
- Weak answer: "There are 76 columns."
- Corrected answer: "It is an interface contract, not just a count."

**[NOTE] Q12. Why add missingness mask?**
- Expected: distinguish real zero from imputed zero; keep stable 152 shape; useful in cross-domain settings.
- Weak answer: "To improve accuracy."
- Corrected answer: "In CICIDS2017 it is constant; its real purpose is robust representation under missing columns."

**[NOTE] Q13. If lab variable does not exist, what happens?**
- Expected: canonical slot is filled with `0.0`, mask slot is `0`, vector remains 152.
- Weak answer: "The model ignores it."
- Corrected answer: "It remains in the vector as an imputed value plus a missingness signal."

**[NOTE] Q14. Why fit scaling only on train?**
- Expected: avoid test statistics leaking into training transformation.
- Weak answer: "Because sklearn says so."
- Corrected answer: "Because preprocessing must be learned only from data available at training time."

**[NOTE] Q15. What if normalization happens before split?**
- Expected: data leakage; inflated metrics; test no longer independent.
- Weak answer: "It may change values."
- Corrected answer: "It contaminates train with test distribution."

### D. Reinforcement Learning
**[NOTE] Q16. What makes this RL?**
- Expected: agent, environment, actions, reward, value learning. Must add limitation: with `gamma=0.0`, it is contextual bandit-like.
- Weak answer: "Because it uses QRDQN."
- Corrected answer: "The interface and learning signal are RL; the temporal structure is not."

**[NOTE] Q17. State/action/reward**
- Expected: state=152-dim flow observation; action=PERMIT/BLOCK; reward=asymmetric cost based on true label and action.
- Weak answer: "State is attack."
- Corrected answer: "The state is what the agent sees, not the ground-truth label."

**[NOTE] Q18. Why offline?**
- Expected: data comes from static dataset/captured CSV; decisions do not interact with live network.
- Weak answer: "Because it runs on a computer."
- Corrected answer: "Offline means no live environment feedback or enforcement."

**[NOTE] Q19. Limitation of static dataset as environment**
- Expected: no causal effect of actions, no adversary reaction, no sequential consequences.
- Weak answer: "It may be smaller."
- Corrected answer: "The structural issue is missing dynamics, not dataset size."

**[NOTE] Q20. Why not autonomous real defense?**
- Expected: no inline blocking, no real-time loop, no deployment, no live state transition.
- Weak answer: "It detects attacks."
- Corrected answer: "Offline detection is not operational enforcement."

### E. Reward And Asymmetric Costs
**[NOTE] Q21. Why is false negative worse?**
- Expected: an attack is permitted, causing potential compromise; false positive blocks legitimate traffic but is usually less severe.
- Weak answer: "Because attacks are bad."
- Corrected answer: "Name the operational consequence: the attacker gets through."

**[NOTE] Q22. How does reward reflect this?**
- Expected: FN `-5.0`, FP `-2.0`, TP `+1.5`, benign permit `0.0`.
- Weak answer: "It penalizes errors."
- Corrected answer: "State exact values and ratio: FN is 2.5x FP magnitude."

**[NOTE] Q23. Risk of over-penalizing false negatives**
- Expected: model may overblock benign traffic; recall high but FPR unacceptable.
- Weak answer: "Better security."
- Corrected answer: "Security has availability/usability cost."

**[NOTE] Q24. High recall but many FPs**
- Expected: catches attacks but causes many false alarms; good for detection sensitivity, bad operationally.
- Weak answer: "It is good."
- Corrected answer: "Assess recall and FPR together."

**[NOTE] Q25. Metric for avoiding missed attacks**
- Expected: attack recall / FNR first, plus FPR and precision to understand cost.
- Weak answer: "Accuracy."
- Corrected answer: "Accuracy hides minority-class failures."

### F. QRDQN
**[NOTE] Q26. Explain QRDQN understandably**
- Expected: DQN variant that learns return distributions via quantiles, not only mean Q-values.
- Weak answer: "A neural network."
- Corrected answer: "Its key feature is distributional value learning."

**[NOTE] Q27. Difference from standard DQN**
- Expected: DQN predicts expected Q; QRDQN predicts quantile distribution and averages for action selection.
- Weak answer: "QRDQN is better."
- Corrected answer: "Say what changes mathematically."

**[NOTE] Q28. Why distributional method might make sense**
- Expected: security decisions involve risk/uncertainty; distribution gives more shape than mean.
- Weak answer: "Because attacks are complex."
- Corrected answer: "Tie it to uncertainty/risk around action returns."

**[NOTE] Q29. Influential hyperparameters**
- Expected: learning rate, batch size, gradient steps, buffer size, learning starts, exploration schedule, `n_quantiles`, `gamma`.
- Weak answer: "All of them."
- Corrected answer: "Name the ones that change learning dynamics and why."

**[NOTE] Q30. Evidence needed to say QRDQN adds value over RF**
- Expected: same protocol, hard splits, multiple seeds, operational metrics, perhaps multi-action/sequential setting.
- Weak answer: "Higher accuracy."
- Corrected answer: "Need robust advantage where RF is not enough."

### G. Random Forest Comparison
**[NOTE] Q31. Why RF is reasonable baseline**
- Expected: strong supervised tabular model, robust, standard, interpretable, no GPU.
- Weak answer: "Because it is simple."
- Corrected answer: "Simple plus strong is why it is a serious baseline."

**[NOTE] Q32. Same protocol comparison**
- Expected: same split, same canonical features, same scaling/preprocessing, same metrics.
- Weak answer: "Run both models."
- Corrected answer: "The comparison is valid only if the data protocol is aligned."

**[NOTE] Q33. If RF is similar or better**
- Expected: interpret honestly; RL is not automatically justified for static binary classification.
- Weak answer: "QRDQN is still better because RL."
- Corrected answer: "If RF wins in the target setting, RF may be preferable."

**[NOTE] Q34. Is QRDQN better because it uses RL?**
- Expected: no.
- Weak answer: "Yes, RL is more advanced."
- Corrected answer: "Algorithm category is not evidence."

**[NOTE] Q35. RF practical advantages**
- Expected: simpler, faster, less compute, easier feature importance, no GPU, stable tabular baseline.
- Weak answer: "None."
- Corrected answer: "RF may be operationally preferable in the easy static setting."

### H. Evaluation And Validation
**[NOTE] Q36. Problem with row-wise random split**
- Expected: similar/duplicate flows can be split across train/test; overestimates generalization.
- Weak answer: "Random is always fair."
- Corrected answer: "Random can be too easy in correlated traffic datasets."

**[NOTE] Q37. What temporal/day split adds**
- Expected: evaluates transfer to different days/attack mixes; harder and more realistic.
- Weak answer: "It changes the dataset."
- Corrected answer: "It tests distribution shift inside CICIDS2017."

**[NOTE] Q38. Leave one CSV out**
- Expected: hold out a complete source file; tests exact-file generalization.
- Weak answer: "More test data."
- Corrected answer: "The unit of separation is the file/domain, not just row count."

**[NOTE] Q39. Domain shift**
- Expected: train and deployment distributions differ; model sees values/patterns outside training support.
- Weak answer: "Different data."
- Corrected answer: "Explain the statistical mismatch and its consequences."

**[NOTE] Q40. Lab validation vs production**
- Expected: lab is controlled, limited, operator-labeled, not enterprise traffic or adversarial production.
- Weak answer: "It is real traffic so production."
- Corrected answer: "Real capture is not the same as production validation."

### I. Critical Discussion
**[NOTE] Q41. Main methodological weakness**
- Expected: static contextual bandit formulation, random split emphasis, Phase 2 domain shift. Pick one and justify.
- Weak answer: "Need better model."
- Corrected answer: "Name a structural weakness."

**[NOTE] Q42. Excessive claim**
- Expected: production readiness, autonomous defense, general QRDQN superiority, sequential RL planning.
- Weak answer: "No excessive claims."
- Corrected answer: "A strong defense includes explicit boundaries."

**[NOTE] Q43. Change to approach real scenario**
- Expected: real-time enforcement loop, multi-action defenses, temporal state, lab calibration, independent labels, latency/throughput tests.
- Weak answer: "Use more data."
- Corrected answer: "Need interaction and operational constraints."

**[NOTE] Q44. What is lost by binarizing labels**
- Expected: attack family/type/severity, multi-class behavior, response granularity.
- Weak answer: "Nothing."
- Corrected answer: "Binary labels simplify but remove response-specific information."

**[NOTE] Q45. Poorly represented attacks/scenarios**
- Expected: slow multi-flow attacks, attacks identified by removed fields, encrypted mimicry, new attacks absent from CICIDS2017.
- Weak answer: "All attacks are represented."
- Corrected answer: "No dataset covers all operational attack behavior."

### J. Anti-Superficial-AI Control Questions
**[NOTE] Q46. Decision you disagree with**
- Strong answer: relying on random split as official headline; it is comparable but optimistic.

**[NOTE] Q47. Weakest part**
- Strong answer: Phase 2, because labels and domain shift are limited and behavior changes by artifact.

**[NOTE] Q48. Limitation not "more data"**
- Strong answer: no causal environment dynamics, `gamma=0.0`, no temporal credit assignment.

**[NOTE] Q49. Redundant section**
- Strong answer: Optuna tuning if presented as central, because main profile was fixed manually and `gamma` search differs from production framing.

**[NOTE] Q50. Result that would change interpretation**
- Strong answer: multi-seed instability, complete hard-split collapse, RF also winning hard splits, or Phase 2 remaining unstable under independent labels.

### Especially Diagnostic Questions
**[NOTE] S1. False positive and false negative example**
- FP: benign flow blocked. Operational cost: legitimate user/service disrupted.
- FN: attack flow permitted. Operational cost: attacker reaches target. Worse under reward.

**[NOTE] S2. Why RL if dataset is labeled/static?**
- Best answer: uses RL interface and reward/value learning, but current `gamma=0.0` makes it a contextual bandit; honest framing is cost-sensitive one-step decision.

**[NOTE] S3. When would RF be methodologically preferable?**
- If the target remains static binary classification, RF matches or beats QRDQN, interpretability and compute matter, and no sequential/multi-action extension is needed.

**[NOTE] S4. Hidden leakage despite obvious drops**
- Duplicates, near-duplicates, split contamination, dataset artifacts, attack schedule correlations, preprocessing fitted on all data.

**[NOTE] S5. What cannot be claimed?**
- Production readiness, active blocking, sequential autonomy, general superiority over RF, solved Phase 2, seed stability.

**[NOTE] S6. "Classification disguised as RL" response**
- Concede the valid part: contextual bandit. Defend the real part: action-value learning with explicit asymmetric reward and QRDQN distributional head.

**[NOTE] S7. Why mask is not minor**
- It keeps input shape stable and tells the model whether zeros are real or imputed. But on CICIDS2017 it is constant, so do not overclaim.

**[NOTE] S8. Another CICIDS2017 CSV vs real traffic**
- Another CSV is intra-dataset shift. Real traffic changes network, services, extractor behavior, normality, and distributions. Much harder.

**[NOTE] S9. More than two defensive actions**
- Reward matrix becomes richer; RL becomes more justified; `gamma>0` may matter if actions affect future states.

**[NOTE] S10. Safety decision that reduced apparent performance**
- Removing ports/IPs/timestamps/Flow IDs to avoid leakage; using hard splits and RF baseline even when they make the story less flattering.

---

## 12. Self-Evaluation Mode: Scoring Rubric ChatGPT Should Use When Quizzing You
**[SPEC]**
ChatGPT should evaluate answers using these dimensions:
- factual correctness.
- project specificity.
- ability to explain mechanism.
- ability to explain "en cristiano".
- limitation awareness.
- artifact awareness.
- no overclaiming.

**[NOTE] 0-5 scoring**
| Score | Meaning | Tutor action |
|---|---|---|
| 5 | Correct, precise, bounded, with project-specific evidence. | Ask a harder follow-up. |
| 4 | Correct but missing one nuance or artifact. | Teach the missing nuance, ask a repair question. |
| 3 | Generic but directionally right. | Demand project-specific details and exact terms. |
| 2 | Buzzword answer; mechanism unclear. | Stop and reteach concept. |
| 1 | Wrong or overclaimed. | Correct strongly, then ask a simpler version. |
| 0 | No answer. | Teach from first principles, then quiz immediately. |

**[NOTE] Required answer pattern**
For each serious question, force this format:

```text
Technical answer:
...

En cristiano:
...

Project-specific evidence:
...

Limitation / what I must not claim:
...
```

**[NOTE] Red flags ChatGPT must challenge**
- "The model blocks attacks in real time."
- "QRDQN is better because RL."
- "The mask improves CICIDS2017 performance."
- "Bootstrap CI proves the model is stable across seeds."
- "Phase 2 proves production readiness."
- "Gamma does not matter at all" without explaining why it does not matter in this static setup.
- "Accuracy is enough."
- "Random split proves generalization."

**[NOTE] Suggested study sequence**
1. Two-minute project explanation.
2. Data pipeline from CSV to 152-vector.
3. Mask and imputation.
4. Anti-leakage.
5. Reward matrix and confusion matrix.
6. Why `gamma=0.0` makes contextual bandit.
7. QRDQN vs DQN.
8. MAIN metrics and bootstrap.
9. RF comparison.
10. Phase 2 and domain shift.
11. Forbidden claims.
12. Mock tribunal.

---

## 13. Glossary: Technical Definition, "En Cristiano", Analogy, And Project-Specific Meaning
**[SPEC]**
This glossary is study material. Use project-specific meanings when answering tutor questions.

**[NOTE] Flow / flujo de red**
- Technical: aggregate statistical record of packets exchanged between endpoints during a communication.
- En cristiano: ficha resumen de una conversacion de red.
- Analogy: resumen bancario instead of every individual movement detail.
- In project: base unit of CICIDS2017 and Phase 2 inference.

**[NOTE] CICIDS2017**
- Technical: public intrusion-detection dataset with flow features exported by CICFlowMeter.
- En cristiano: banco de ejemplos de trafico benigno y ataques.
- Analogy: cuaderno de ejercicios de ciberseguridad.
- In project: primary training/evaluation dataset, not proof of production readiness.

**[NOTE] Canonical schema**
- Technical: fixed ordered feature interface independent of original column names.
- En cristiano: plantilla fija de 76 medidas.
- Analogy: formulario con las mismas casillas para cualquier fuente.
- In project: `FEATURES_CANON` in `src/canonical_schema.py`.

**[NOTE] Missingness mask**
- Technical: binary vector marking whether each canonical value was present/valid or imputed.
- En cristiano: lista de "este dato es real" vs "este dato lo rellene".
- Analogy: asterisco next to uncertain values in a spreadsheet.
- In project: 76 extra dimensions; 152 total observation.

**[NOTE] Imputation**
- Technical: replacing missing or invalid values with a default value.
- En cristiano: rellenar huecos.
- Analogy: putting `0` in an empty form field but marking it as filled by us.
- In project: default `0.0`; mask records missingness.

**[NOTE] Data leakage / fuga de informacion**
- Technical: evaluation contamination or feature shortcut that gives access to information unavailable in real deployment.
- En cristiano: el modelo ve pistas que no deberia.
- Analogy: examen con respuestas escritas en el margen.
- In project: avoided by dropping IPs, timestamps, Flow IDs and direct ports; also checked with shuffled labels.

**[NOTE] StandardScaler / z-score**
- Technical: subtract train mean and divide by train standard deviation per feature.
- En cristiano: poner variables en escala comparable.
- Analogy: convertir notas de distintas escalas a una escala comun.
- In project: fitted only on train to avoid leakage.

**[NOTE] Train/test split**
- Technical: partition used to train and evaluate independently.
- En cristiano: estudiar con una parte y examinarse con otra.
- Analogy: practice problems vs final exam.
- In project: default random stratified 80/20 seed 42; hard day split for Check C.

**[NOTE] Stratified split**
- Technical: preserves class proportions in train and test.
- En cristiano: reparte benignos y ataques manteniendo proporciones.
- Analogy: dividir caramelos de dos sabores sin cambiar mezcla.
- In project: default random split.

**[NOTE] Domain shift**
- Technical: train and evaluation/deployment distributions differ.
- En cristiano: el mundo del examen no se parece al mundo de entrenamiento.
- Analogy: entrenar conduccion en ciudad y examinarse en nieve.
- In project: main Phase 2 risk.

**[NOTE] BENIGN / ATTACK labels**
- Technical: binary target labels.
- En cristiano: trafico normal vs malicioso.
- Analogy: alarma no activa vs amenaza.
- In project: `0=BENIGN`, `1=ATTACK`.

**[NOTE] PERMIT / BLOCK actions**
- Technical: action space decisions.
- En cristiano: dejar pasar o bloquear.
- Analogy: portero que deja entrar o no.
- In project: experimental/offline actions, not real firewall enforcement.

**[NOTE] True positive (TP)**
- Technical: attack correctly blocked.
- En cristiano: paro un ataque real.
- Analogy: alarma suena cuando hay incendio.
- In project: reward `+1.5`.

**[NOTE] False positive (FP)**
- Technical: benign flow blocked.
- En cristiano: bloqueo trafico bueno.
- Analogy: alarma suena sin incendio.
- In project: reward `-2.0`.

**[NOTE] False negative (FN)**
- Technical: attack permitted.
- En cristiano: dejo pasar un ataque.
- Analogy: alarma no suena durante incendio.
- In project: reward `-5.0`, most severe error.

**[NOTE] True negative / omission**
- Technical: benign flow permitted.
- En cristiano: dejo pasar trafico bueno.
- Analogy: alarma callada cuando no hay incendio.
- In project: called `omission`, reward `0.0`.

**[NOTE] Accuracy**
- Technical: proportion of correct predictions.
- En cristiano: porcentaje total de aciertos.
- Analogy: nota global.
- In project: useful but insufficient under class imbalance.

**[NOTE] Precision attack**
- Technical: among predicted attacks/blocks, proportion truly attack.
- En cristiano: de lo que bloqueo como ataque, cuanto era ataque.
- Analogy: cuanto pescado hay en la red y no basura.
- In project: MAIN `0.97378`.

**[NOTE] Recall attack**
- Technical: among real attacks, proportion detected/blocked.
- En cristiano: cuantos ataques reales atrapo.
- Analogy: cuantos incendios detecta la alarma.
- In project: MAIN `0.99536`; key for missed attacks.

**[NOTE] F1 attack**
- Technical: harmonic mean of precision and recall for attack class.
- En cristiano: equilibrio entre cazar ataques y no bloquear demasiado benigno.
- Analogy: nota combinada de sensibilidad y limpieza.
- In project: MAIN `0.98445`; RF random `0.99676`.

**[NOTE] FPR**
- Technical: false positive rate; benign flows wrongly blocked.
- En cristiano: tasa de falsas alarmas.
- Analogy: usuarios legitimos molestados.
- In project: MAIN point `0.00658`.

**[NOTE] FNR**
- Technical: false negative rate; attacks missed.
- En cristiano: tasa de ataques que se cuelan.
- Analogy: incendios no detectados.
- In project: MAIN point `0.00465`.

**[NOTE] Balanced accuracy**
- Technical: average recall across classes.
- En cristiano: acierto medio tratando las clases de forma mas justa.
- Analogy: nota media por asignatura, no por numero de preguntas.
- In project: MAIN `0.99439`.

**[NOTE] MCC**
- Technical: Matthews correlation coefficient; global classification quality using all confusion-matrix cells.
- En cristiano: nota robusta incluso con clases desbalanceadas.
- Analogy: evaluacion que no se deja enganar por acertar solo la clase grande.
- In project: MAIN `0.98068`.

**[NOTE] Bootstrap confidence interval**
- Technical: interval estimated by resampling test data with replacement.
- En cristiano: repetir muchas simulaciones del test para ver cuanto baila la metrica.
- Analogy: sondeo repetido sobre la misma muestra.
- In project: measures uncertainty for one trained model, not seed variability.

**[NOTE] Reinforcement learning (RL)**
- Technical: learning from interaction through states, actions and rewards.
- En cristiano: aprender por premios y castigos.
- Analogy: entrenar una estrategia con consecuencias.
- In project: used as framework, but current setup is one-step contextual bandit.

**[NOTE] Contextual bandit**
- Technical: one-step decision problem conditioned on context; action does not affect future state.
- En cristiano: veo una situacion, decido, cobro premio/castigo, y se acaba.
- Analogy: elegir una opcion en una maquina con pista contextual.
- In project: honest description under `gamma=0.0`.

**[NOTE] MDP**
- Technical: Markov Decision Process with state transitions influenced by actions.
- En cristiano: mundo donde mis acciones cambian lo que pasa despues.
- Analogy: ajedrez: cada jugada cambia el tablero.
- In project: formal environment interface, but practical dynamics are static.

**[NOTE] Gamma / discount factor**
- Technical: weight assigned to future rewards.
- En cristiano: cuanto me importa el futuro.
- Analogy: valorar dinero futuro vs dinero ahora.
- In project: `gamma=0.0`, so only immediate reward matters.

**[NOTE] DQN**
- Technical: Deep Q-Network estimating action values with a neural network.
- En cristiano: red que puntua cada accion.
- Analogy: tabla mental aproximada por una red neuronal.
- In project: base idea behind QRDQN.

**[NOTE] QRDQN**
- Technical: distributional DQN using quantile regression to model return distributions.
- En cristiano: no solo aprende la media; aprende el reparto de posibles premios.
- Analogy: conocer toda la campana de notas, no solo la media.
- In project: main algorithm from `sb3_contrib`.

**[NOTE] Quantile**
- Technical: point dividing a distribution into probability regions.
- En cristiano: marca que corta una distribucion.
- Analogy: percentiles de altura.
- In project: 200 quantiles in main profile.

**[NOTE] Replay buffer**
- Technical: memory of past transitions sampled for off-policy learning.
- En cristiano: cuaderno de experiencias pasadas.
- Analogy: repasar jugadas guardadas.
- In project: buffer size `1_000_000` in main profile.

**[NOTE] Off-policy**
- Technical: can learn from data generated by older/different behavior policies.
- En cristiano: aprende de experiencias guardadas, no solo de lo que acaba de hacer.
- Analogy: estudiar partidas grabadas.
- In project: QRDQN off-policy learning from replay buffer.

**[NOTE] Target network**
- Technical: delayed/copy network used to stabilize targets.
- En cristiano: copia congelada para no perseguir una diana que se mueve todo el tiempo.
- Analogy: apuntar a una diana fija durante un rato.
- In project: exists, but future target contribution is neutralized by `gamma=0.0`.

**[NOTE] Epsilon-greedy**
- Technical: exploration strategy choosing random action with probability epsilon.
- En cristiano: a veces prueba al azar, a veces elige lo mejor que cree.
- Analogy: probar plato nuevo de vez en cuando.
- In project: epsilon goes `1.0 -> 0.02` in main profile.

**[NOTE] Quantile-Huber loss**
- Technical: loss for learning quantile return distribution robustly.
- En cristiano: regla de error para ajustar cuantiles sin volverse loca con errores grandes.
- Analogy: corregir examenes ponderando pasarse o quedarse corto.
- In project: QRDQN training objective.

**[NOTE] Random Forest**
- Technical: ensemble of decision trees.
- En cristiano: muchos arboles votando.
- Analogy: comite de expertos simples.
- In project: serious supervised baseline; wins random split, collapses on day split.

**[NOTE] RUN_ID**
- Technical: unique run identifier for experiment traceability.
- En cristiano: matricula del experimento.
- Analogy: numero de expediente.
- In project: essential for citing metrics.

**[NOTE] Phase 2**
- Technical: offline inference over extracted flow CSVs from private lab captures.
- En cristiano: probar el modelo sobre flujos capturados en mi laboratorio, sin bloquear en vivo.
- Analogy: simulacro con datos propios, no despliegue real.
- In project: current open risk due to domain shift.

**[NOTE] Artifact-backed result**
- Technical: metric supported by committed run output/config/metrics.
- En cristiano: cifra que puedo rastrear a un archivo.
- Analogy: dato con recibo.
- In project: required for honest documentation.

---

## 14. Changelog
**[SPEC]**
- `1.0.1` on 2026-06-29:
  - Added explicit guidance for ChatGPT to use GitHub app/repo access to verify current code, docs, and committed artifacts.
  - Added conflict rule: live repository code plus committed artifacts override this packet if they disagree.

**[SPEC]**
- `1.0.0` on 2026-06-29:
  - Created single-file Spanish study packet for ChatGPT tutoring.
  - Included HADS-style AI reading instruction.
  - Consolidated project overview, architecture, invariants, data pipeline, RL framing, QRDQN, training, results, limitations, question bank, self-evaluation rubric, and glossary.
  - Marked critical caveats around `gamma=0.0`, Random Forest, Phase 2, and production claims.
