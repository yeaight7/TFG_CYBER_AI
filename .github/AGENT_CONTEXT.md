# AGENT_CONTEXT (TFG_CYBER_AI)

## Objetivo del TFG

Desarrollar un **agente defensor de ciberseguridad basado en Aprendizaje por Refuerzo (Reinforcement Learning)** capaz de clasificar tráfico de red como benigno o ataque, y decidir acciones óptimas (PERMIT/BLOCK) para maximizar la seguridad minimizando falsos positivos y negativos.

### Fases del Proyecto

**Fase 1 — Clasificación/Detección sobre Datasets**:
- Entrenar agente RL sobre datasets históricos y modernos
- Comparar con baselines supervisados (Random Forest)
- Definir esquema canónico de features
- Validar funcionamiento del framework RL

**Fase 2 — Entorno Simulado con Tráfico Generado**:
- Generar tráfico real (benigno + ataques) con herramientas (Kali, scripts)
- Capturar tráfico (PCAP)
- Extraer features de flujos (flow features) en tiempo real
- Agente RL decide PERMIT/BLOCK sobre tráfico capturado
- Almacenar dataset de simulación (features + acción + label ground-truth)

---

## Cambio de Trayectoria (Decisión Fundamental)

**IMPORTANTE**: El proyecto adoptó un cambio de trayectoria crítico para garantizar coherencia y reproducibilidad.

### Problema Original

Entrenar con múltiples datasets que tienen **columnas/features diferentes** crea varios problemas:
1. El modelo RL necesita un espacio de observación **fijo** (misma longitud, mismo significado por posición)
2. Si entrenamos con `[a,b,c,d,e]` y luego en simulación tenemos `[a,b,c,x,y]`, el modelo falla
3. Comparaciones entre datasets son injustas (bias según features disponibles)
4. En simulación, no sabríamos qué features extraer del tráfico real

### Solución: Esquema Canónico de Features (FEATURES_CANON)

Definir un **conjunto fijo de features canónicas** que:
1. Sirva de "lenguaje común" para TODOS los datasets
2. Sea extraíble de tráfico real/PCAP en simulación (flow-based features)
3. Permita entrenar el agente con múltiples datasets de forma justa
4. Garantice que el agente ve siempre el mismo espacio de observación

**Todos los datasets deben convertirse al esquema canónico mediante adapters.**

---

## Regla de "Justicia" (Fairness)

Para garantizar comparaciones válidas entre modelos y datasets:

- ✅ **Mismo conjunto de features canónicas** (mismo orden, mismo significado)
- ✅ **Mismo preprocesado** (manejo de missingness, scaling)
- ✅ **Mismas métricas y costes** (FP/FN, sistema de recompensas)
- ✅ **Misma semilla y estrategia de split** (reproducibilidad)

**Consecuencia**: No podemos simplemente concatenar datasets con columnas diferentes. Debemos mapear primero al esquema canónico.

---

## Datasets

El proyecto está diseñado para trabajar con **MÚLTIPLES datasets** (no solo 2). Cada dataset pasa por un **adapter** que lo convierte al esquema canónico.

### Datasets Actuales

#### 1. NSL-KDD (Benchmark Histórico)
- **Kaggle**: `hassan06/nslkdd`
- **Propósito**: Benchmark histórico para Fase 1, demostración del framework RL
- **Características**: 
  - 41 features originales (basadas en conexiones antiguas)
  - Features muy diferentes a datasets modernos flow-based
  - Dataset de 1999, tráfico simulado de laboratorio
- **Uso en el proyecto**: 
  - ✅ Fase 1: Warm-up, comparación con literatura, prueba del framework
  - ❌ **NO** forma parte del modelo final para simulación
  - ❌ **NO** se usa para definir el esquema canónico
- **Razón**: Sus features son incompatibles con flow extractors modernos y no se pueden extraer de PCAP en simulación

#### 2. CICIDS2017 (Dataset Principal Moderno)
- **Kaggle**: `chethuhn/network-intrusion-dataset`
- **Propósito**: Dataset principal que **define el esquema canónico**
- **Características**:
  - ~80 features flow-based (extraídas con CICFlowMeter)
  - Tráfico moderno (2017), incluye ataques web, botnets, DDoS, etc.
  - 2.8M+ muestras
- **Uso en el proyecto**:
  - ✅ Base para definir `FEATURES_CANON`
  - ✅ Entrenamiento de modelos para Fase 2
  - ✅ Referencia para transición a simulación

#### 3. Datasets Futuros (Extensibilidad)

El diseño soporta añadir **N datasets adicionales** mediante adapters:

- **Candidatos ideales**: Familia CIC (CICIDS2018, CSE-CIC-IDS2018), UNSW-NB15, otros con flow features
- **Requisito**: Que sus features sean compatibles o mapeables al esquema canónico
- **Proceso**: Crear `load_<nombre_dataset>.py` que mapee al esquema canónico + máscara de missingness

**Nota importante**: El valor del multi-dataset training es mayor cuando los datasets comparten el mismo **tipo de features** (flow-based). Mezclar datasets incompatibles (como NSL-KDD con CICIDS) requiere mapeos artificiales que pueden introducir ruido.

---

## Features Canónicas (FEATURES_CANON)

### Estado Actual: Definidas Formalmente ✅

**76 features canónicas** definidas en `src/canonical_schema.py`, seleccionadas de las columnas de CICIDS2017 (CICFlowMeter output).

Criterios de selección:

1. **Existen en CICIDS2017** (dataset principal moderno)
2. **Extraíbles de tráfico real/PCAP** en simulación mediante flow extractors (CICFlowMeter, Zeek, etc.)
3. **Sin data leakage**: NO incluir IPs, timestamps absolutos, Flow IDs, puertos específicos
4. **Preferiblemente numéricas**: Facilita el aprendizaje RL
5. **Estables y robustas**: No dependen de peculiaridades del dataset

### Categorías de Features Canónicas

Las 76 features se organizan en las siguientes categorías:

- **Estadísticas generales del flujo** (5): flow_duration, total_fwd/bwd_packets, total_length_of_fwd/bwd_packets
- **Estadísticas de tamaño de paquetes forward** (4): fwd_packet_length_{max, min, mean, std}
- **Estadísticas de tamaño de paquetes backward** (4): bwd_packet_length_{max, min, mean, std}
- **Tasas de flujo** (2): flow_bytes_per_s, flow_packets_per_s
- **Inter-arrival times del flujo** (4): flow_iat_{mean, std, max, min}
- **Inter-arrival times forward** (5): fwd_iat_{total, mean, std, max, min}
- **Inter-arrival times backward** (5): bwd_iat_{total, mean, std, max, min}
- **Flags TCP** (12): fwd/bwd_psh_flags, fwd/bwd_urg_flags, fin/syn/rst/psh/ack/urg/cwe/ece_flag_count
- **Header length** (2): fwd/bwd_header_length
- **Paquetes por segundo** (2): fwd/bwd_packets_per_s
- **Estadísticas de longitud global** (5): min/max/mean/std/variance_packet_length
- **Ratios y derivadas** (4): down_up_ratio, average_packet_size, avg_fwd/bwd_segment_size
- **Bulk statistics** (6): fwd/bwd_avg_bytes/packets_per_bulk, fwd/bwd_avg_bulk_rate
- **Sub-flow statistics** (4): subflow_fwd/bwd_packets, subflow_fwd/bwd_bytes
- **Ventana TCP** (2): init_win_bytes_forward/backward
- **Datos activos** (2): act_data_pkt_fwd, min_seg_size_forward
- **Tiempos activos/idle** (8): active/idle_{mean, std, max, min}

### Implementación

- **Archivo**: `src/canonical_schema.py`
- **Lista**: `FEATURES_CANON` — lista ordenada de 76 nombres canónicos
- **Mappings**: `CICIDS2017_TO_CANON` (76/76 presentes), `NSL_KDD_TO_CANON` (3/76 presentes)
- **Función**: `map_to_canonical(df, column_mapping)` — devuelve `CanonicalResult` con X, mask, combined
- **Dimensión del vector de observación**: 152 (76 features + 76 máscara de missingness)

### Máscara de Missingness (Missingness Mask)

**Implementada** en `src/canonical_schema.py`:

- Features ausentes se imputan con `DEFAULT_IMPUTATION_VALUE = 0.0`
- Se marca con máscara de missingness: `m_i = 0` (0 = ausente, 1 = presente)
- Valores NaN/Inf en features presentes se imputan y se marcan como ausentes

**Vector de observación final**:
```
obs = [x_1, x_2, ..., x_76, m_1, m_2, ..., m_76]   # 152 dimensiones
```

Donde:
- `x_i` = valor de la feature i (imputado si no existe)
- `m_i` = 1 si la feature estaba presente, 0 si fue imputada

Esto permite al agente RL **aprender qué features son confiables** para cada tipo de dato.

---

## Pipeline de Simulación (Visión Fase 2)

```
┌─────────────────────────────────────────────────────────────┐
│ 1. GENERACIÓN DE TRÁFICO                                    │
│    - Kali Linux con herramientas de ataque                  │
│    - Scripts de tráfico benigno (navegación, descargas)     │
│    - Ataques: DDoS, port scans, exploits, etc.              │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. CAPTURA DE TRÁFICO (PCAP)                                │
│    - tcpdump / Wireshark / tshark                           │
│    - Captura de paquetes en tiempo real                     │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. EXTRACCIÓN DE FEATURES (Flow Features)                   │
│    - CICFlowMeter / Zeek / custom extractor                 │
│    - Produce vectores en formato FEATURES_CANON             │
│    - Output: [x_1, ..., x_d, m_1, ..., m_d]                │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. AGENTE RL DECIDE (PERMIT/BLOCK)                          │
│    - Lee vector de features del flujo                       │
│    - Acción 0 = PERMIT (permitir)                           │
│    - Acción 1 = BLOCK (bloquear)                            │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. ALMACENAR DATASET DE SIMULACIÓN                          │
│    - CSV/Parquet con: features + acción + label truth      │
│    - Label truth: conocemos qué tráfico generamos (0/1)     │
│    - Permite evaluar decisiones del agente                  │
└─────────────────────────────────────────────────────────────┘
```

### Pregunta Pendiente de Diseño

**¿Qué feature extractor usar en simulación?**
- Opción A: **CICFlowMeter-like** (mismo que CICIDS2017, compatibilidad directa)
- Opción B: **Zeek-like** (más flexible, ampliamente usado en industria)
- Opción C: **Custom extractor** (máximo control, más trabajo)

Decisión: **Pendiente de evaluación en Fase 2**. Probablemente CICFlowMeter para máxima compatibilidad con CICIDS2017.

---

## Estado Actual del Proyecto (Implementado)

### Archivos y Componentes Existentes

#### `src/canonical_schema.py`
- ✅ Definición formal de `FEATURES_CANON` con 76 features canónicas
- ✅ Mappings: `CICIDS2017_TO_CANON` (76/76), `NSL_KDD_TO_CANON` (3/76)
- ✅ Función `map_to_canonical()` con máscara de missingness
- ✅ `CanonicalResult` dataclass con X, mask, combined, feature_names
- ✅ Vector de observación: 152 dimensiones (76 features + 76 máscara)

#### `src/load_nsl_kdd.py`
- ✅ Loader para NSL-KDD desde Kaggle (kagglehub)
- ✅ Preprocesamiento: one-hot encoding (legacy) o esquema canónico
- ✅ Retorna: `(X_train, y_train, X_test, y_test, scaler, feature_names)`
- ✅ Soporte esquema canónico con `use_canonical=True` (3/76 features mapeadas)
- ✅ Modo legacy compatible con `use_canonical=False`

#### `src/load_cicids2017.py`
- ✅ Loader para CICIDS2017 desde Kaggle
- ✅ Limpieza de datos: eliminación de IPs, timestamps, Flow IDs
- ✅ Manejo de NaNs e infinitos
- ✅ Split estratificado y split por día (CSV-split)
- ✅ Helper público para listar los 8 CSVs reales de CICIDS2017 en orden determinista
- ✅ Split exacto por nombre de CSV (`load_cicids2017_exact_csv_split`) para validaciones leave-one-CSV-out
- ✅ Retorna: `(X_train, y_train, X_test, y_test, scaler, feature_names)`
- ✅ Soporte esquema canónico con `use_canonical=True` (76/76 features mapeadas)
- ✅ API unificada `load_cicids2017_split()` con modo random y day

#### `src/rl_defender_env.py`
- ✅ Entorno RL custom (Gymnasium)
- ✅ Espacio de observación: vector de features (Box, 152 dims con esquema canónico)
- ✅ Espacio de acciones: Discreto(2) — 0=PERMIT, 1=BLOCK
- ✅ Sistema de recompensas configurable (TP/FP/FN/TN)
- ✅ Soporte para shuffle y episodios limitados

#### `src/train_rl_defender.py`
- ✅ Script de entrenamiento QRDQN con Stable-Baselines3 / sb3-contrib
- ✅ Generación automática de RUN_ID con timestamp
- ✅ Logging a TensorBoard (`runs/<dataset>/<RUN_ID>/`)
- ✅ Evaluación en test con confusion matrix y classification report
- ✅ Guardado de modelos en `models/<RUN_ID>.zip`
- ✅ CLI con argparse (--smoke, --preset, --split-mode, --timesteps, --max-rows)
- ✅ Persistencia de scaler y train_percentiles (para inferencia Phase 2)
- ✅ Auto-detección de GPU/CPU

#### `src/validate_checks.py`
- ✅ Framework de validación con Checks A, B, C
- ✅ Check A: evaluación directa model.predict vs y_test
- ✅ Check B: shuffled-labels anti-leakage test
- ✅ Check C: CSV-split day generalization (train Mon-Wed, test Thu-Fri)

#### `src/validate_leave_one_csv_out.py`
- ✅ Validación separada leave-one-exact-CSV-out sobre los 8 CSVs reales de CICIDS2017
- ✅ Entrena QRDQN una vez por fold dejando 1 CSV exacto como test y 7 como train
- ✅ Persistencia en `runs/validation/VAL_leave_one_csv_out_<timestamp>/`
- ✅ Métricas por fold + agregados globales (accuracy, balanced accuracy, specificity, FPR/FNR, reward)

#### `src/tune_hparams.py`
- ✅ Optimización de hiperparámetros con Optuna
- ✅ Grid search de learning_rate, batch_size, gradient_steps, gamma, train_freq

#### `src/scaling_utils.py`
- ✅ Utilidades de escalado de features (StandardScaler helpers)

#### `src/baseline_random_forest.py`
- ✅ Baseline supervisado con Random Forest
- ✅ Evaluación comparable con métricas de clasificación
- ✅ Guardado de modelo con RUN_ID en `models/<RUN_ID>.joblib`

#### `scripts/predict_real_traffic.py` (v1, legacy)
- ✅ Script de inferencia Phase 2 sobre flows extraídos de PCAPs
- ⚠️ Producía predicciones extremas (all-block o all-allow)

#### `scripts/predict_real_traffic_v2.py` (v2, robusta)
- ✅ Pipeline de inferencia Phase 2 robusto
- ✅ Z-score clipping para manejar distribución shift
- ✅ Soporte de percentile clipping (features raw) y z-score clipping (scaled)
- ✅ CLI con argparse (--flows, --model, --scaler, --percentiles, --clip-z)
- ✅ Guardado de config.json, metrics.json, diagnostics.json, predictions.csv

#### `lab/docker/`
- ✅ Docker Compose para lab privado (nginx target + generador de tráfico)
- ✅ Red aislada (internal: true, sin acceso a internet)

### Modelos Entrenados

| Modelo | Tipo | Dataset | Accuracy |
|--------|------|---------|----------|
| C01 smoke (.zip) | QRDQN | CICIDS2017 50k | 0.9697 |
| C01 full (.zip) | QRDQN | CICIDS2017 250k | 0.9962 |
| C02 fast (.zip) | QRDQN | CICIDS2017 100k | 0.9766 |
| **C03 full (.zip)** | **QRDQN** | **CICIDS2017 500k** | **0.9986** |
| rf_nslkdd.joblib | Random Forest | NSL-KDD | 0.7693 |
| A01, A02 (.zip) | DQN | NSL-KDD | Phase 1 ablation |
| rl_defender_dqn.zip | DQN | NSL-KDD | Early prototype |

### Directorios y Estructura

```
TFG_CYBER_AI/
├── datasets/          — CICIDS2017 descargado automáticamente
├── docs/              — Documentación, contexto, decisiones
├── experiments/       — Tracking de experimentos
├── lab/               — Docker Compose para lab privado
│   └── docker/        — Compose file + generador de tráfico
├── models/            — Modelos guardados (.zip, .joblib)
├── pcaps/             — PCAPs capturados y CSVs de flows extraídos
├── runs/              — Resultados con RUN_ID (TensorBoard logs)
│   ├── cicids2017/    — Runs entrenamiento QRDQN (C01, C02, C03)
│   ├── validation/    — Runs validation checks (A, B, C)
│   ├── phase2/        — Runs inferencia Phase 2 (P2, P2v2)
│   ├── optuna/        — Estudios de hiperparámetros
│   └── nslkdd/        — Runs Phase 1 (NSL-KDD benchmark)
├── scripts/           — Scripts de inferencia Phase 2
│   ├── predict_real_traffic.py     — v1 (legacy)
│   └── predict_real_traffic_v2.py  — v2 (robusta)
└── src/               — Código fuente Python
    ├── canonical_schema.py      — Esquema canónico de features
    ├── load_cicids2017.py       — Adapter CICIDS2017
    ├── load_nsl_kdd.py          — Adapter NSL-KDD
    ├── rl_defender_env.py       — Entorno RL (Gymnasium)
    ├── train_rl_defender.py     — Entrenamiento QRDQN
    ├── validate_checks.py       — Checks A/B/C de validación
    ├── tune_hparams.py          — Optuna hyperparameter tuning
    ├── scaling_utils.py         — Utilidades de escalado
    └── baseline_random_forest.py — Baseline Random Forest
```

---

## Próximos Pasos Inmediatos

### 1. Auditoría de Columnas CICIDS2017

- [x] Ejecutar `load_cicids2017.py` y listar todas las columnas disponibles
- [x] Clasificar columnas en: flow features, metadata, labels
- [x] Identificar columnas de leakage (ya eliminadas, verificar)
- [x] Documentar significado de cada feature

### 2. Definir FEATURES_CANON Formalmente

- [x] Seleccionar ~30-80 features flow-based de CICIDS2017 → **76 features seleccionadas**
- [x] Documentar criterio de selección para cada feature
- [x] Crear lista `FEATURES_CANON` en código → `src/canonical_schema.py`
- [x] Verificar que todas son extraíbles de PCAP

### 3. Implementar Adapters con Esquema Canónico

- [x] Modificar `load_cicids2017.py` para mapear a `FEATURES_CANON`
- [x] Implementar máscara de missingness
- [x] Modificar `load_nsl_kdd.py` para mapear a `FEATURES_CANON` (con modo legacy)
- [x] Vector final: `[x_1..x_76, m_1..m_76]` → 152 dimensiones

### 4. Entrenar Modelos con Esquema Canónico

- [x] Entrenar QRDQN sobre CICIDS2017 con esquema canónico (C01 smoke, C01 full)
- [x] Entrenar con más datos (C02 fast 100k, C03 full 500k) → **Mejor modelo: C03 full, accuracy 0.9986**
- [x] Validar métricas (Check A: accuracy 0.9939, Check B: no leakage, Check C: day-split 0.8414)
- [x] Implementar validación leave-one-exact-CSV-out por CSV real de CICIDS2017
- [x] Documentar resultados en `docs/results.md`

### 5. Optimización de Hiperparámetros

- [x] Optuna study con 10 trials → mejor accuracy 0.9939 (lr=5.2e-4, batch=256)
- [x] Resultados en `runs/optuna/study_20260212_222134.json`

### 6. Inferencia Phase 2 (Lab Traffic)

- [x] Implementar script v1 de inferencia (`scripts/predict_real_traffic.py`)
- [x] Implementar script v2 robusto con z-clipping (`scripts/predict_real_traffic_v2.py`)
- [x] Ejecutar inferencia sobre PCAPs capturados en lab privado
- [ ] Calibrar/fine-tune para reducir FP en tráfico benigno real (distribución shift detectado)

### 7. Preparar para Multi-Dataset Training

- [x] Validar que adapters producen el mismo formato de salida (152 dims)
- [ ] Implementar función de "merge" de datasets con esquema canónico
- [x] Verificar que la máscara de missingness funciona correctamente
- [ ] Experimentar con entrenamiento combinado (si hay múltiples datasets disponibles)

---

## Decisiones Técnicas Pendientes

### 1. Feature Extractor para Simulación
- **Opciones**: CICFlowMeter, Zeek, custom
- **Decisión**: Pendiente de Fase 2
- **Recomendación actual**: CICFlowMeter (compatibilidad directa con CICIDS2017)

### 2. Estrategia de Imputación en Máscara de Missingness
- **Opciones**: 0, media, mediana, forward-fill, modelo de imputación
- **Decisión**: **0 para todas las features** (`DEFAULT_IMPUTATION_VALUE = 0.0`)
- **Implementación**: `src/canonical_schema.py`
- **Nota**: Se puede personalizar por feature en el futuro si es necesario

### 3. Número Final de Features Canónicas
- **Rango**: 30-80 features
- **Decisión**: **76 features canónicas** (definidas en `src/canonical_schema.py`)
- **Trade-off**: 76 features proporcionan cobertura completa de estadísticas de flujo CICFlowMeter

### 4. Inclusión de NSL-KDD en Modelo Final
- **Opciones**: Incluir con adapter, excluir del modelo final
- **Decisión**: **Excluir del modelo final para simulación**
- **Razón**: Features incompatibles con flow extractors modernos
- **Uso**: Solo como benchmark histórico en Fase 1

---

## Referencias y Recursos

- **Repositorio**: https://github.com/yeaight7/TFG_CYBER_AI
- **Datasets**:
  - NSL-KDD: `hassan06/nslkdd` (Kaggle)
  - CICIDS2017: `chethuhn/network-intrusion-dataset` (Kaggle)
- **Frameworks**:
  - RL: Stable-Baselines3, sb3-contrib (QRDQN)
  - Entorno: Gymnasium
  - ML: scikit-learn
  - Hyperparameter tuning: Optuna
- **Documentación interna**:
  - `.github/copilot-instructions.md`: Convenciones de código
  - `AGENTS.md`: Checklist para coding agents

---

## Notas Finales

Este documento es la **fuente de verdad** del proyecto. Si hay contradicciones con otros documentos, este prevalece.

Actualiza este documento cuando:
- Se defina formalmente `FEATURES_CANON`
- Se implementen adapters al esquema canónico
- Se completen pasos de "Próximos pasos inmediatos"
- Cambien decisiones de diseño fundamentales
- Se añadan nuevos datasets

**Última actualización**: 2026-04-11 (actualización de estado: validación leave-one-exact-CSV-out añadida, sin modificar resultados históricos existentes)
